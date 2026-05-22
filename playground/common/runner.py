"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from brax.training import networks as brax_networks
from brax.training import distribution as brax_dist
from brax.training import types as brax_types
from flax import linen
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

from playground.common.export_onnx import export_onnx


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        if jax.default_backend() == "gpu":
            jax.config.update(
                "jax_persistent_cache_enable_xla_caches",
                "xla_gpu_per_fusion_autotune_cache_dir",
            )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        print("-----------")
        print(
            f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
        )
        print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        export_onnx(
            params,
            self.action_size,
            self.ppo_params,
            self.obs_size,  # may not work
            output_path=onnx_export_path,
            layer_norm=getattr(self.args, 'layer_norm', False),
        )

        # Policy quality eval - logs to TB so the dashboard shows actual capability
        # rather than the noisy alive-dominated training reward
        try:
            import sys as _sys, os as _os
            _proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
            if _proj_root not in _sys.path:
                _sys.path.insert(0, _proj_root)

            env_name = getattr(self.args, 'env', 'joystick')
            if env_name == 'standing':
                from analysis.quick_standing_eval import assess_standing
                eval_metrics = assess_standing(onnx_export_path)
                print(f"  Standing HEADLINE: {eval_metrics.get('standing/HEADLINE', 0):.3f}  "
                      f"stillness: {eval_metrics.get('standing/stillness_score', 0):.3f}  "
                      f"push: {eval_metrics.get('standing/push_score', 0):.3f}")
            else:
                from analysis.quick_walking_eval import assess_walking
                eval_metrics = assess_walking(onnx_export_path)
                print(f"  Walking HEADLINE: {eval_metrics.get('walking/HEADLINE', 0):.3f}  "
                      f"walking: {eval_metrics.get('walking/WALKING_SCORE', 0):.3f}  "
                      f"responsive: {eval_metrics.get('walking/responsiveness_avg', 0):.3f}")

            for k, v in eval_metrics.items():
                self.writer.add_scalar(k, float(v), current_step)
        except Exception as exc:
            print(f"  [eval skipped: {exc}]")

    @staticmethod
    def _make_ppo_networks_with_layernorm(
        observation_size,
        action_size,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        activation=linen.swish,
        policy_obs_key='state',
        value_obs_key='state',
        **kwargs,
    ):
        parametric_action_distribution = brax_dist.NormalTanhDistribution(
            event_size=action_size
        )
        policy_network = brax_networks.make_policy_network(
            parametric_action_distribution.param_size,
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=activation,
            layer_norm=True,
            obs_key=policy_obs_key,
        )
        value_network = brax_networks.make_value_network(
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            obs_key=value_obs_key,
        )
        return ppo_networks.PPONetworks(
            policy_network=policy_network,
            value_network=value_network,
            parametric_action_distribution=parametric_action_distribution,
        )

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20

        use_layer_norm = getattr(self.args, 'layer_norm', False)
        policy_layers = getattr(self.args, 'policy_layers', None)
        activation_name = getattr(self.args, 'activation', 'swish')
        activation_fn = linen.elu if activation_name == 'elu' else linen.swish

        layer_overrides = {}
        if policy_layers:
            layer_overrides['policy_hidden_layer_sizes'] = tuple(policy_layers)
            layer_overrides['value_hidden_layer_sizes'] = tuple(policy_layers)

        if "network_factory" in self.ppo_params:
            if use_layer_norm:
                nf_kwargs = dict(self.ppo_params.network_factory)
                nf_kwargs.update(layer_overrides)
                nf_kwargs['activation'] = activation_fn
                network_factory = functools.partial(
                    self._make_ppo_networks_with_layernorm, **nf_kwargs
                )
            else:
                nf_kwargs = dict(self.ppo_params.network_factory)
                nf_kwargs.update(layer_overrides)
                nf_kwargs['activation'] = activation_fn
                network_factory = functools.partial(
                    ppo_networks.make_ppo_networks, **nf_kwargs
                )
            del self.ppo_training_params["network_factory"]
        else:
            if use_layer_norm:
                network_factory = functools.partial(
                    self._make_ppo_networks_with_layernorm,
                    activation=activation_fn, **layer_overrides
                )
            else:
                network_factory = functools.partial(
                    ppo_networks.make_ppo_networks,
                    activation=activation_fn, **layer_overrides
                ) if layer_overrides or activation_name != 'swish' else ppo_networks.make_ppo_networks

        if use_layer_norm:
            print("*** LayerNorm ENABLED on policy network ***")
        if policy_layers:
            print(f"*** Network layers: {tuple(policy_layers)} ***")
        if activation_name != 'swish':
            print(f"*** Activation: {activation_name} ***")

        use_adaptive_kl = getattr(self.args, 'adaptive_kl', False)
        if use_adaptive_kl:
            from brax.training.agents.ppo import optimizer as ppo_optimizer
            self.ppo_training_params['learning_rate_schedule'] = ppo_optimizer.LRSchedule.ADAPTIVE_KL
            self.ppo_training_params['desired_kl'] = 0.01
            print("*** Adaptive KL LR schedule ENABLED (desired_kl=0.01) ***")
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        self.ppo_training_params["clipping_epsilon_value"] = 0.2

        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
