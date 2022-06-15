# coding=utf-8


import numpy as np
import jax
import jax.numpy as jnp

import warnings

# from tf_sparse.utils import merge_duplicated_sparse_index
import numbers

from jax.experimental.sparse import BCOO
from jax.experimental.sparse._base import JAXSparse

"""
Sparse Matrix for Computation
"""


def _segment_softmax(data, segment_ids, num_segments):
    max_values = tf.math.unsorted_segment_max(data, segment_ids, num_segments=num_segments)
    gathered_max_values = tf.gather(max_values, segment_ids)
    exp = tf.exp(data - tf.stop_gradient(gathered_max_values))
    denominator = tf.math.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tf.gather(denominator, segment_ids)
    score = exp / gathered_denominator
    return score


# class SparseMatrixSpec(TypeSpec):
#
#     __slots__ = ["_shape", "_dtype"]
#
#     value_type = property(lambda self: SparseMatrix)
#
#     def __init__(self, shape=None, dtype=dtypes.float32):
#         """Constructs a type specification for a `SparseMatrix`.
#
#         Args:
#           shape: The dense shape of the `tf_sparse.SparseMatrix`, or `None` to allow any dense
#             shape.
#           dtype: `tf.DType` of values in the `SparseMatrix`.
#         """
#         self._shape = tensor_shape.as_shape(shape)
#         self._dtype = dtypes.as_dtype(dtype)
#
#     def _serialize(self):
#         return (self._shape, self._dtype)
#
#     @property
#     def dtype(self):
#         """The `tf.dtypes.DType` specified by this type for the SparseTensor."""
#         return self._dtype
#
#     @property
#     def shape(self):
#         """The `tf.TensorShape` specified by this type for the SparseTensor."""
#         return self._shape
#
#     @property
#     def _component_specs(self):
#         rank = self._shape.ndims
#         num_values = None
#         return [
#             tensor_spec.TensorSpec([num_values, rank], dtypes.int64),
#             tensor_spec.TensorSpec([num_values], self._dtype),
#             tensor_spec.TensorSpec([rank], dtypes.int64),
#             tensor_spec.TensorSpec([], dtypes.bool)
#         ]
#
#     def _to_components(self, value):
#         return [value.index, value.value, tf.convert_to_tensor(value._shape), tf.convert_to_tensor(value.is_diag)]
#
#     def _from_components(self, tensor_list):
#         return SparseMatrix(tensor_list[0], tensor_list[1], shape=tensor_list[2], is_diag=tensor_list[3])


class SparseMatrix(JAXSparse):

    # https://stackoverflow.com/questions/40252765/overriding-other-rmul-with-your-classs-mul
    __array_priority__ = 10000

    def __init__(self, index, data=None, shape=None, merge=False, is_diag=False):
        """
        Sparse Matrix for efficient computation.
        :param index:
        :param data:
        :param shape: [num_rows, num_cols], shape of the adjacency matrix.
        :param merge: Whether to merge duplicated edge
        """

        super().__init__(None, shape=shape)

        self.index = SparseMatrix.cast_index(index)
        if data is None:
            self.data = jnp.ones([self.index.shape[1]], dtype=jnp.float64)
        else:
            self.data = SparseMatrix.cast_value(data)
        self.is_diag = is_diag

        # if value is not None:
        #     self.value = SparseMatrix.cast_value(value)
        # else:
        #     if index_is_tensor:
        #         num_values = tf.shape(self.index)[1]
        #         value = tf.ones([num_values], dtype=tf.float32)
        #     else:
        #         num_values = np.shape(self.index)[1]
        #         value = np.ones([num_values], dtype=np.float32)
        #     self.value = value

        # if merge:
        #     self.index, [self.value] = merge_duplicated_sparse_index(self.index, [self.value],
        #                                                              merge_modes=["sum"])


        # self._shape = SparseMatrix.cast_shape(shape)
        # # self._shape = tf.convert_to_tensor(shape)
        # self.shape = tensor_util.constant_value_as_shape(self._shape)
        #
        # self.is_diag = tf.convert_to_tensor(is_diag)



    @classmethod
    def cast_index(cls, index):
        if not isinstance(index, jnp.ndarray):
            index = jnp.array(index)
        # index = tf.cast(index, tf.int32)
        return index


    @classmethod
    def cast_value(cls, value):
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value)
        # value = tf.cast(value, dtype=tf.float32)
        return value

        # if isinstance(value, list):
        #     value = np.array(value).astype(np.float32)
        # elif isinstance(value, np.ndarray):
        #     value = value.astype(np.float32)
        # elif tf.is_tensor(value):
        #     value = tf.cast(value, tf.float32)
        # return value

    # @classmethod
    # def cast_shape(cls, shape):
    #     return tf.cast(tf.convert_to_tensor(shape), tf.int32)

    @property
    def row(self):
        return self.index[0]

    @property
    def col(self):
        return self.index[1]

    def __len__(self):
        return self.shape[0]

    def with_value(self, new_value):
        return self.__class__(self.index, new_value, shape=self.shape, is_diag=self.is_diag)

    # def merge_duplicated_index(self):
    #     edge_index, [edge_weight] = merge_duplicated_sparse_index(self.index, [self.value], merge_modes=["sum"])
    #     return self.__class__(edge_index, edge_weight, shape=self._shape, is_diag=False)

    def negative(self):
        return self.__class__(
            index=self.index,
            data=-self.data,
            shape=self.shape,
            is_diag=self.is_diag
        )

    def __neg__(self):
        return self.negative()

    def tree_flatten(self):
        pass

    def transpose(self):
        row, col = self.index[0], self.index[1]
        transposed_index = jnp.stack([col, row], axis=0)
        return self.__class__(transposed_index, value=self.data, shape=[self.shape[1], self.shape[0]], is_diag=self.is_diag)

    def map_value(self, map_func):
        return self.__class__(self.index, map_func(self.data), self.shape, is_diag=self.is_diag)

    def _segment_reduce(self, segment_func, axis=None, keepdims=False):

        # reduce by row
        if axis == -1 or axis == 1:
            reduce_axis = 0
        # reduce by col
        elif axis == 0 or axis == -2:
            reduce_axis = 1
        else:
            raise Exception("Invalid axis value: {}, axis shoud be -1, -2, 0, or 1".format(axis))

        reduce_index = self.index[reduce_axis]
        num_reduced = self.shape[reduce_axis]

        output = segment_func(self.data, reduce_index, num_reduced)
        if keepdims:
            output = jnp.expand_dims(output, axis=axis)

        return output

    def segment_sum(self, axis=None, keepdims=False):
        return self._segment_reduce(jax.ops.segment_sum, axis=axis, keepdims=keepdims)

    # def segment_mean(self, axis=None, keepdims=False):
    #     return self._segment_reduce(, axis=axis, keepdims=keepdims)

    def segment_max(self, axis=None, keepdims=False):
        return self._segment_reduce(jax.ops.segment_max, axis=axis, keepdims=keepdims)

    def segment_min(self, axis=None, keepdims=False):
        return self._segment_reduce(jax.ops.segment_min, axis=axis, keepdims=keepdims)

    def segment_softmax(self, axis=-1):

        # reduce by row
        if axis == -1 or axis == 1:
            reduce_index = self.index[0]
            num_reduced = self.shape[0]
        # reduce by col
        elif axis == 0 or axis == -2:
            reduce_index = self.index[1]
            num_reduced = self.shape[1]
        else:
            raise Exception("Invalid axis value: {}, axis shoud be -1, -2, 0, or 1".format(axis))

        normed_value = _segment_softmax(self.data, reduce_index, num_reduced)

        return self.__class__(self.index, normed_value, shape=self.shape)

    # https://stackoverflow.com/questions/45731484/tensorflow-how-to-perform-element-wise-multiplication-between-two-sparse-matrix
    def _mul_sparse_matrix(self, other):
        a, b = self, other
        ab = ((a + b).map_value(jnp.square) - a.map_value(jnp.square) - b.map_value(jnp.square)).map_value(
            lambda x: x * 0.5)
        return ab

    def _rmul_sparse_matrix(self, other):
        return self._mul_sparse_matrix(other)

    def _mul_scalar(self, scalar):
        return self.__class__(self.index, self.data * scalar, self._shape)

    def _rmul_scalar(self, scalar):
        return self._mul_scalar(scalar)

    def _mul_dense(self, dense):
        sparse_tensor = self.to_sparse_tensor()
        return self.__class__.from_sparse_tensor(sparse_tensor * dense)

    def __mul__(self, other):

        if isinstance(other, SparseMatrix):
            return self._mul_sparse_matrix(other)
        else:
            if isinstance(other, numbers.Number) or len(other.shape) == 0:
                return self._mul_scalar(other)
            else:
                return self._mul_dense(other)
            # return tf.cond(
            #     tf.rank(other) == 0,
            #     lambda: self._mul_scalar(other),  # if other is scalar
            #     lambda: self._mul_dense(other)  # if other is dense
            # )

        # if is scalar
        # elif tf.rank(other) == 0:
        #     return self._mul_scalar(other)
        # elif tf.is_tensor(other) or isinstance(other, np.ndarray):
        #     return self._mul_dense(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    # def _matmul_dense(self, h):
    #     row, col = self.index[0], self.index[1]
    #     repeated_h = tf.gather(h, col)
    #     if self.value is not None:
    #         repeated_h *= tf.expand_dims(self.value, axis=-1)
    #     reduced_h = tf.math.unsorted_segment_sum(repeated_h, row, num_segments=self._shape[0])
    #     return reduced_h

    def _matmul_dense(self, h):
        row, col = self.index
        repeated_h = h[col]
        weighted_repeated_h = repeated_h * jnp.expand_dims(self.data, axis=1)
        output = jax.ops.segment_sum(weighted_repeated_h, row, self.shape[0])
        return output

    # def _matmul_sparse(self, other):
    #
    #     if self.is_diag:
    #         return other.rmatmul_diag(self.value)
    #     elif other.is_diag:
    #         return self.matmul_diag(other.value)
    #
    #     warnings.warn("The operation \"SparseMatrix @ SparseMatrix\" does not support gradient computation.")
    #
    #     csr_matrix_a = self._to_csr_sparse_matrix()
    #     csr_matrix_b = other._to_csr_sparse_matrix()
    #
    #     csr_matrix_c = sparse_matrix_sparse_mat_mul(
    #         a=csr_matrix_a, b=csr_matrix_b, type=self.value.dtype
    #     )
    #
    #     sparse_tensor_c = csr_sparse_matrix_to_sparse_tensor(
    #         csr_matrix_c, type=self.value.dtype
    #     )
    #
    #     output_class = self.__class__ if issubclass(self.__class__, other.__class__) else other.__class__
    #
    #     return output_class.from_sparse_tensor(sparse_tensor_c, merge=True)

    # sparse_adj @ other
    def matmul(self, other):
        if isinstance(other, SparseMatrix):
            return self._matmul_sparse(other)
        else:
            return self._matmul_dense(other)

    def _rmatmul_dense(self, h):
        transposed_h = jnp.transpose(h)
        # sparse_adj' @ h'
        transpoed_output = self.transpose() @ transposed_h
        # h @ sparse_adj = (sparse_adj' @ h')'
        output = jnp.transpose(transpoed_output)
        return output

    # other @ sparse_adj
    def rmatmul(self, other):
        if isinstance(other, SparseMatrix):
            return other._matmul_sparse(self)
        else:
            return self._rmatmul_dense(other)

    # # other_sparse_adj @ sparse_adj
    # def rmatmul_sparse(self, other):
    #     # h'
    #     transposed_other = other.transpose()
    #     # sparse_adj' @ h'
    #     transpoed_output = self.transpose() @ transposed_other
    #     # h @ sparse_adj = (sparse_adj' @ h')'
    #     output = transpoed_output.transpose()
    #     return output

    # self @ diagonal_matrix
    def matmul_diag(self, diagonals):
        col = self.index[1]
        updated_edge_weight = self.data * tf.gather(diagonals, col)
        return self.__class__(self.index, updated_edge_weight, self._shape)

    # self @ diagonal_matrix
    def rmatmul_diag(self, diagonals):
        row = self.index[0]
        updated_edge_weight = tf.gather(diagonals, row) * self.data
        return self.__class__(self.index, updated_edge_weight, self._shape)

    # self @ other (other is a dense tensor or SparseAdj)
    def __matmul__(self, other):
        return self.matmul(other)

    # h @ self (h is a dense tensor)
    def __rmatmul__(self, other):
        return self.rmatmul(other)

    def add_diag(self, diagonals):
        if tf.rank(diagonals) == 0:
            diagonals = tf.fill([self.shape[0]], diagonals)

        diagonal_matrix = self.__class__.from_diagonals(diagonals)

        return diagonal_matrix + self

    def remove_diag(self):
        row, col = self.index[0], self.index[1]
        mask = tf.not_equal(row, col)

        masked_index = tf.boolean_mask(self.index, mask)
        masked_value = tf.boolean_mask(self.data, mask)

        return self.__class__(masked_index, masked_value, self._shape)

    def set_diag(self, diagonals):
        return self.remove_diag().add_diag(diagonals)

    def eliminate_zeros(self):
        # edge_index_is_tensor = tf.is_tensor(self.index)
        # edge_weight_is_tensor = tf.is_tensor(self.value)

        mask = tf.not_equal(self.data, 0.0)
        masked_edge_index = tf.boolean_mask(self.index, mask, axis=1)
        masked_edge_weight = tf.boolean_mask(self.data, mask)

        # if not edge_index_is_tensor:
        #     masked_edge_index = masked_edge_index.numpy()
        #
        # if not edge_weight_is_tensor:
        #     masked_edge_weight = masked_edge_weight.numpy()

        return self.__class__(masked_edge_index, masked_edge_weight, shape=self._shape)

    def merge(self, other, merge_mode):
        """
        element-wise merge

        :param other:
        :param merge_mode:
        :return:
        """

        # edge_index_is_tensor = tf.is_tensor(self.index)
        # edge_weight_is_tensor = tf.is_tensor(self.value)

        combined_index = tf.concat([self.index, other.index], axis=1)
        combined_value = tf.concat([self.data, other.data], axis=0)

        merged_index, [merged_value] = merge_duplicated_sparse_index(
            combined_index, props=[combined_value], merge_modes=[merge_mode])

        # if tf.executing_eagerly() and not edge_index_is_tensor:
        #     merged_index = merged_index.numpy()
        #
        # if tf.executing_eagerly() and not edge_weight_is_tensor:
        #     merged_value = merged_value.numpy()

        output_class = self.__class__ if issubclass(self.__class__, other.__class__) else other.__class__
        merged_sparse_adj = output_class(merged_index, merged_value, shape=self._shape)

        return merged_sparse_adj

    def __add__(self, other_sparse_adj):
        """
        element-wise sparse adj addition: self + other_sparse_adj
        :param other_sparse_adj:
        :return:
        """
        return self.merge(other_sparse_adj, merge_mode="sum")

    # def __radd__(self, other_sparse_adj):
    #     """
    #     element-wise sparse adj addition: other_sparse_adj + self
    #     :param other_sparse_adj:
    #     :return:
    #     """
    #     return other_sparse_adj + self

    def __sub__(self, other):
        return self + other.negative()

    # def __rsub__(self, other):
    #     return other.__sub__(self)

    def dropout(self, drop_rate, training=False):
        if training and drop_rate > 0.0:
            output_value = tf.compat.v2.nn.dropout(self.data, drop_rate)
        else:
            output_value = self.data
        return self.with_value(output_value)

    # def add_self_loop(self, fill_weight=1.0):
    #     num_nodes = self.shape[0]
    #     updated_edge_index, updated_edge_weight = add_self_loop_edge(self.index, num_nodes,
    #                                                                  edge_weight=self.value,
    #                                                                  fill_weight=fill_weight)
    #     return SparseMatrix(updated_edge_index, updated_edge_weight, self.shape)
    #
    # def remove_self_loop(self):
    #     updated_edge_index, updated_edge_weight = remove_self_loop_edge(self.index, edge_weight=self.value)
    #     return SparseMatrix(updated_edge_index, updated_edge_weight, self.shape)

    def _to_csr_sparse_matrix(self):

        return sparse_tensor_to_csr_sparse_matrix(
            indices=tf.cast(tf.transpose(self.index, [1, 0]), tf.int64),
            values=self.data,
            dense_shape=tf.cast(self._shape, dtype=tf.int64)
        )

    @classmethod
    def from_diagonals(cls, diagonals):
        """
        Construct a SparseAdj from diagonals
        :param diagonals:
        :return:
        """
        num_rows = tf.shape(diagonals)[0]
        row = tf.range(0, num_rows, dtype=tf.int32)
        col = row
        index = tf.stack([row, col], axis=0)
        return cls(index, diagonals, shape=[num_rows, num_rows], is_diag=True)

    @classmethod
    def eye(cls, num_rows):
        """
        Construct a SparseAdj with ones on diagonal
        :param diagonals:
        :return:
        """
        diagonals = tf.ones([num_rows], dtype=tf.float32)
        return cls.from_diagonals(diagonals)

    # @classmethod
    # def from_sparse_tensor(cls, sparse_tensor: tf.sparse.SparseTensor, merge=False):
    #     return cls(
    #         index=tf.transpose(sparse_tensor.indices, [1, 0]),
    #         value=sparse_tensor.values,
    #         shape=sparse_tensor.dense_shape,
    #         merge=merge
    #     )

    def to_sparse_tensor(self):

        sparse_tensor = tf.sparse.SparseTensor(
            indices=tf.cast(tf.transpose(self.index, [1, 0]), tf.int64),
            values=self.data,
            dense_shape=tf.cast(self._shape, tf.int64)
        )
        # sparse_tensor = tf.sparse.reorder(sparse_tensor)
        return sparse_tensor

    def to_dense(self):
        sparse_tensor = self.to_sparse_tensor()
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        return tf.sparse.to_dense(sparse_tensor)

    def __str__(self):
        return "SparseMatrix: \n" \
               "index => \n" \
               "{}\n" \
               "value => {}\n" \
               "shape => {}\n" \
               "is_diag => {}".format(self.index, self.data, self._shape, self.is_diag)

    def __repr__(self):
        return self.__str__()