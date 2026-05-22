"""Compatibility shim for geoms_colliding from old mujoco_playground.

mujoco_playground >= 0.2.0 removed the collision module in favour of sensor-based
contact detection via <touch> sensors in the MJCF. Our envs rely on the older
contact-pair lookup pattern, so this helper preserves that interface.
"""
import jax
import jax.numpy as jp


def geoms_colliding(data, geom_id_1: int, geom_id_2: int) -> jax.Array:
    """Return True if a contact between the two geoms is active in `data`.

    Replicates `mujoco_playground._src.collision.geoms_colliding` from playground 0.0.5.
    """
    geom1 = data.contact.geom1
    geom2 = data.contact.geom2
    forward = (geom1 == geom_id_1) & (geom2 == geom_id_2)
    reverse = (geom1 == geom_id_2) & (geom2 == geom_id_1)
    return jp.any(forward | reverse)
