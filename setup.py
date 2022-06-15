# coding=utf-8

from setuptools import setup, find_packages

setup(
    name="jax_sparse",
    python_requires='>3.5.0',
    version="0.0.1",
    author="Jun Hu",
    author_email="hujunxianligong@gmail.com",
    packages=find_packages(
        exclude=[
            'benchmarks',
            'data',
            'demo',
            'dist',
            'doc',
            'docs',
            'logs',
            'models',
            'test'
        ]
    ),
    install_requires=[
        "jax >= 0.3.0",
        # "networkx >= 2.1",
        # "scipy >= 1.1.0",
        # "scikit-learn >= 0.22",
        # "ogb_lite >= 0.0.3",
        # "tqdm"
    ],
    extras_require={

    },
    description="Efficient and Friendly Sparse Matrix Library for JAX.",
    license="GNU General Public License v3.0 (See LICENSE)",
    long_description="Efficient and Friendly Sparse Matrix Library for JAX.",
    # long_description=open("README.rst", "r", encoding="utf-8").read(),
    url="https://github.com/CrawlScript/jax-sparse"
)