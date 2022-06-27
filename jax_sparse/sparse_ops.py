# coding=utf-8

from .sparse_matrix import SparseMatrix


# Construct a SparseAdj from diagonals
def diags(diagonals):
    return SparseMatrix.from_diagonals(diagonals)


# Construct a SparseAdj with ones on diagonal
def eye(num_nodes):
    return SparseMatrix.eye(num_nodes)