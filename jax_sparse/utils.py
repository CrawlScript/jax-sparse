# coding=utf-8
import jax.numpy as jnp
import jax.ops



def convert_sparse_index_to_hash(index, hash_key=None):


    index = jnp.asarray(index, dtype=jnp.int64)

    # if not edge_index_is_tensor or edge_index.dtype != tf.int64:
    #     edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int64)

    if hash_key is None:
        hash_key = jnp.max(index) + 1
    else:
        hash_key = jnp.int64(hash_key)

    row, col = index[0], index[1]

    hash = hash_key * row + col

    return hash, hash_key


def convert_hash_to_sparse_index(hash, hash_key):


    # if not edge_hash_is_tensor:
    #     edge_hash = tf.convert_to_tensor(edge_hash)

    hash = jnp.asarray(hash, dtype=jnp.int64)
    hash_key = jnp.int64(hash_key)

    row = jnp.floor_divide(hash, hash_key)
    col = jnp.mod(hash, hash_key)

    edge_index = jnp.stack([row, col], axis=0)

    return edge_index


def merge_duplicated_sparse_index(sparse_index, props=None, merge_modes=None):
    """
    merge_modes: list of merge_mode ("min", "max", "mean", "sum")
    """

    if props is not None:
        if type(merge_modes) is not list:
            raise Exception("type error: merge_modes should be a list of strings")
        if merge_modes is None:
            merge_modes = ["sum"] * len(props)


    hash, hash_key = convert_sparse_index_to_hash(sparse_index)
    unique_hash, unique_index = jnp.unique(hash, return_inverse=True)
    # mask =

    unique_sparse_index = convert_hash_to_sparse_index(unique_hash, hash_key)

    if props is None:
        unique_props = None
    else:
        unique_props = []
        for prop, merge_mode in zip(props, merge_modes):

            if prop is None:
                unique_prop = None
            else:

                prop = jnp.asarray(prop)

                if merge_mode == "min":
                    merge_func = jax.ops.segment_min
                elif merge_mode == "max":
                    merge_func = jax.ops.segment_max
                elif merge_mode == "sum":
                    merge_func = jax.ops.segment_sum
                else:
                    raise Exception("wrong merge mode: {}".format(merge_mode))
                unique_prop = merge_func(prop, unique_index, jnp.shape(unique_hash)[0])

            unique_props.append(unique_prop)

    return unique_sparse_index, unique_props


# def convert_sparse_index_to_upper(sparse_index, props=None, merge_modes=None):
#     """
#
#     :param sparse_index:
#     :param props:
#     :param merge_modes: List of merge modes. Merge Modes: "min" | "max" | "mean" | "sum"
#     :return:
#     """
#
#     edge_index_is_tensor = tf.is_tensor(sparse_index)
#
#     if not edge_index_is_tensor:
#         sparse_index = tf.convert_to_tensor(sparse_index, dtype=tf.int32)
#
#     row = tf.math.reduce_min(sparse_index, axis=0)
#     col = tf.math.reduce_max(sparse_index, axis=0)
#
#     upper_edge_index = tf.stack([row, col], axis=0)
#     upper_edge_index, upper_edge_props = merge_duplicated_sparse_index(upper_edge_index, props, merge_modes)
#
#     if not edge_index_is_tensor:
#         upper_edge_index = upper_edge_index.numpy()
#
#     return upper_edge_index, upper_edge_props

