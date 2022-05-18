import os

from setuptools import find_packages, setup


def _load_requirements(path):
    with open(path, encoding="utf-8") as requirements:
        return [requirement.strip() for requirement in requirements]


base_dir = os.path.dirname(os.path.abspath(__file__))
install_requires = _load_requirements(os.path.join(base_dir, "requirements.txt"))
tests_require = _load_requirements(os.path.join(base_dir, "test", "requirements.txt"))

setup(
    name="nmt-wizard-docker",
    version="0.1.0",
    license="MIT",
    description="Dockerization of NMT frameworks",
    author="OpenNMT",
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "tests": tests_require,
    },
    packages=find_packages(include=["nmtwizard"]),
)
