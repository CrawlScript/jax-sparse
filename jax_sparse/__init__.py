# coding=utf-8

from jax.config import config
config.update("jax_enable_x64", True)

from .sparse_matrix import SparseMatrix
from .sparse_ops import diags, eye