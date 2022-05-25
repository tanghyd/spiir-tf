from setuptools import setup, find_packages

setup(
    name="spiir-tensorflow",
    version="0.0.1",
    packages=find_packages(),
    package_dir={"": "src"},
    description="A work-in-progress Python package for SPIIR's TensorFlow research and development.",
    author="Daniel Tang",
    author_email="daniel.tang@uwa.edu.au",
)
