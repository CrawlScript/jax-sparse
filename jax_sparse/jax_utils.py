# # coding=utf-8
# import jax.numpy as jnp
# import jax.ops
#
#
# def jnp_unique(value, fill_value=-1):
#     value = jnp.asarray(value)
#     unique_value, unique_index = jnp.unique(value, return_inverse=True, size=value.shape[0], fill_value=fill_value)
#
#     mask = unique_value != fill_value
#     unique_value = unique_value[mask]
#     unique_index = unique_index[mask]
#     return unique_value, unique_index
