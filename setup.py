import setuptools

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dopamax",
    version="0.2.1",
    author="Ryan Strauss",
    author_email="ryanrstrauss@icloud.com",
    description="Reinforcement learning in pure JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rystrauss/dopamax",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    platforms="any",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "dopamax=dopamax._scripts.cli:cli",
        ],
    },
    install_requires=[
        "jax>=0.4.35",
        "jaxlib>=0.4.35",
        "chex>=0.1.87",
        "brax>=0.11.0",
        "click>=8.1.7",
        "distrax>=0.1.5",
        "dm-env>=1.6",
        "dm-haiku>=0.0.13",
        "pgx>=2.4.2",
        "einops>=0.8.0",
        "ffmpeg>=1.4",
        "imageio>=2.25.1",
        "mctx>=0.0.5",
        "moviepy>=1.0.3",
        "ml-collections>=0.1.1",
        "optax>=0.2.4",
        "pygame>=2.6.1",
        "numpy>=1.22.4",
        "rlax>=0.1.6",
        "tqdm>=4.64.1",
        "wandb>=0.18.7",
        "flashbax>=0.1.2",
    ],
    tests_require=[
        "pytest",
        "gymnasium==1.0.0",
    ],
)
