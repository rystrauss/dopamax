try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dopamax",
    version="0.1.0",
    author="Ryan Strauss",
    author_email="ryanrstrauss@icloud.com",
    description="Reinforcement learning in pure JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rystrauss/dopamax",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["dopamax"],
    include_package_data=True,
    zip_safe=False,
    platforms="any",
    python_requires=">=3.10",
    install_requires=[
        "jax",
        "chex",
        "brax",
        "click",
        "distrax",
        "dm-env",
        "dm-haiku",
        "einops",
        "ffmpeg",
        "moviepy",
        "imageio",
        "pygame",
        "ml_collections",
        "numpy",
        "rlax",
        "tqdm",
        "wandb",
    ],
    tests_require=[
        "pytest",
        "gymnasium",
    ],
)
