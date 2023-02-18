import setuptools

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(CURRENT_DIR, "requirements.txt")) as f:
    requirements = f.read().splitlines()

with open(os.path.join(CURRENT_DIR, "test-requirements.txt")) as f:
    test_requirements = f.read().splitlines()

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
    packages=setuptools.find_packages(),
    include_package_data=True,
    platforms="any",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "dopamax=dopamax._scripts.cli:cli",
        ],
    },
    install_requires=requirements,
    tests_require=test_requirements,
)
