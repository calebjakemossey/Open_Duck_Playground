
import numpy as np
import onnx
from onnx import helper, TensorProto


def export_onnx(
    params, act_size, ppo_params, obs_size, output_path="ONNX.onnx",
    layer_norm=False,
):
    print(" === EXPORT ONNX === ")

    norm = params[0]
    mean = np.array(norm.mean["state"] if hasattr(norm, 'mean') else norm["mean"]["state"], dtype=np.float32)
    std = np.array(norm.std["state"] if hasattr(norm, 'std') else norm["std"]["state"], dtype=np.float32)

    p1 = params[1]
    policy = getattr(p1, 'policy', p1)
    jax_params = policy.get('params') if isinstance(policy, dict) else policy
    if jax_params is None:
        print("ERROR: Could not extract policy params")
        return

    hidden_sizes = list(ppo_params.network_factory.policy_hidden_layer_sizes)
    final_size = act_size * 2
    layer_sizes = hidden_sizes + [final_size]

    kernels = []
    biases = []
    ln_scales = []
    ln_biases = []
    for i in range(len(layer_sizes)):
        layer_key = f"hidden_{i}"
        k = np.array(jax_params[layer_key]["kernel"], dtype=np.float32)
        b = np.array(jax_params[layer_key]["bias"], dtype=np.float32)
        kernels.append(k)
        biases.append(b)
        print(f"  Layer {layer_key}: kernel {k.shape}, bias {b.shape}")

        if layer_norm and i < len(layer_sizes) - 1:
            ln_key = f"LayerNorm_{i}"
            if ln_key in jax_params:
                s = np.array(jax_params[ln_key]["scale"], dtype=np.float32)
                lb = np.array(jax_params[ln_key]["bias"], dtype=np.float32)
                ln_scales.append(s)
                ln_biases.append(lb)
                print(f"  LayerNorm {ln_key}: scale {s.shape}, bias {lb.shape}")
            else:
                ln_scales.append(None)
                ln_biases.append(None)

    nodes = []
    initializers = []

    initializers.append(helper.make_tensor("mean", TensorProto.FLOAT, mean.shape, mean.flatten()))
    initializers.append(helper.make_tensor("std", TensorProto.FLOAT, std.shape, std.flatten()))

    nodes.append(helper.make_node("Sub", ["obs", "mean"], ["normed_sub"]))
    nodes.append(helper.make_node("Div", ["normed_sub", "std"], ["normed"]))

    prev_output = "normed"
    for i in range(len(layer_sizes)):
        k_name = f"kernel_{i}"
        b_name = f"bias_{i}"
        mm_out = f"matmul_{i}"
        add_out = f"dense_{i}"

        initializers.append(helper.make_tensor(k_name, TensorProto.FLOAT, kernels[i].shape, kernels[i].flatten()))
        initializers.append(helper.make_tensor(b_name, TensorProto.FLOAT, biases[i].shape, biases[i].flatten()))

        nodes.append(helper.make_node("MatMul", [prev_output, k_name], [mm_out]))
        nodes.append(helper.make_node("Add", [mm_out, b_name], [add_out]))

        is_last = (i == len(layer_sizes) - 1)

        if not is_last:
            if layer_norm and i < len(ln_scales) and ln_scales[i] is not None:
                ln_s_name = f"ln_scale_{i}"
                ln_b_name = f"ln_bias_{i}"
                ln_out = f"ln_{i}"
                initializers.append(helper.make_tensor(ln_s_name, TensorProto.FLOAT, ln_scales[i].shape, ln_scales[i].flatten()))
                initializers.append(helper.make_tensor(ln_b_name, TensorProto.FLOAT, ln_biases[i].shape, ln_biases[i].flatten()))

                eps_name = f"ln_eps_{i}"
                initializers.append(helper.make_tensor(eps_name, TensorProto.FLOAT, [], [1e-5]))

                mean_out = f"ln_mean_{i}"
                nodes.append(helper.make_node("ReduceMean", [add_out], [mean_out], axes=[-1], keepdims=1))

                centered = f"ln_centered_{i}"
                nodes.append(helper.make_node("Sub", [add_out, mean_out], [centered]))

                sq = f"ln_sq_{i}"
                nodes.append(helper.make_node("Mul", [centered, centered], [sq]))

                var_out = f"ln_var_{i}"
                nodes.append(helper.make_node("ReduceMean", [sq], [var_out], axes=[-1], keepdims=1))

                var_eps = f"ln_var_eps_{i}"
                nodes.append(helper.make_node("Add", [var_out, eps_name], [var_eps]))

                inv_std = f"ln_invstd_{i}"
                nodes.append(helper.make_node("Sqrt", [var_eps], [f"ln_std_{i}"]))
                nodes.append(helper.make_node("Div", [centered, f"ln_std_{i}"], [inv_std]))

                scaled = f"ln_scaled_{i}"
                nodes.append(helper.make_node("Mul", [inv_std, ln_s_name], [scaled]))
                nodes.append(helper.make_node("Add", [scaled, ln_b_name], [ln_out]))

                swish_in = ln_out
            else:
                swish_in = add_out

            sig_out = f"sigmoid_{i}"
            swish_out = f"swish_{i}"
            nodes.append(helper.make_node("Sigmoid", [swish_in], [sig_out]))
            nodes.append(helper.make_node("Mul", [swish_in, sig_out], [swish_out]))
            prev_output = swish_out
        else:
            prev_output = add_out

    half = final_size // 2
    nodes.append(helper.make_node("Split", [prev_output], ["loc", "scale_raw"], axis=-1, split=[half, half]))
    nodes.append(helper.make_node("Tanh", ["loc"], ["continuous_actions"]))

    input_tensor = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_size])
    output_tensor = helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, half])

    graph = helper.make_graph(nodes, "policy", [input_tensor], [output_tensor], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    onnx.save(model, "ONNX.onnx")

    test_input = np.ones((1, obs_size), dtype=np.float32)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        result = sess.run(None, {"obs": test_input})
        print(f"  ONNX prediction (verified): {result[0][0][:5]}...")
    except ImportError:
        print("  (onnxruntime not installed - skipping verification)")

    print(f"  Saved: {output_path}")
