import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# This vendored copy is installed from FastWAM-NavSIM with --no-deps.
# Keep requirements.txt in the repository as upstream documentation, but do not
# expose its old pins as install_requires metadata. Those pins include versions
# such as torch==2.0.1 and numpy==1.23.4, which conflict with the tested
# FastWAM-NavSIM environment. The repository-level setup script installs the
# compatible locked environment explicitly.
requirements = []

# Installs
setuptools.setup(
    name="navsim",
    version="1.1.0",
    author="University of Tuebingen",
    author_email="kashyap.chitta@uni-tuebingen.de",
    description="NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking",
    url="https://github.com/autonomousvision/navsim",
    python_requires=">=3.9",
    packages=setuptools.find_packages(script_folder),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
