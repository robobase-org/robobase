import codecs
import os
from pathlib import Path

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


core_requirements = [
    "torch>1.13",
    "moviepy",
    "natsort",
    "omegaconf",
    "hydra-core",
    "hydra-joblib-launcher",
    # Fix for solver_iter before 1.0.0
    "gymnasium @ git+https://git@github.com/stepjam/Gymnasium.git@0.29.2",
    "wandb<=0.15.12",
    "termcolor",
    "opencv-python-headless",
    "numpy<2",
    "imageio",
    "timm",
    "scipy",
    "einops",
    "diffusers==0.29.0",
]

setuptools.setup(
    version=get_version("robobase/__init__.py"),
    name="robobase",
    author="robobase",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=core_requirements,
    package_data={
        "": [str(p.resolve()) for p in Path("robobase/cfgs/").glob("**/*.yaml")]
    },
    extras_require={
        "dev": ["pre-commit", "pytest", "mvp @ git+https://github.com/ir413/mvp"],
        "dmc": [
            "dm_control",
        ],
        "rlbench": [
            "rlbench @ git+https://git@github.com/stepjam/RLBench.git@b80e51feb3694d9959cb8c0408cd385001b01382",
        ],
        "bigym": [
            "bigym @ git+https://github.com/chernyadev/bigym.git",
        ],
        "d4rl": [
            "d4rl @ git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl",
            "gym",
            "cython<3",
        ],
    },
)
