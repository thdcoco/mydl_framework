# setup.py

from setuptools import setup, find_packages

setup(
    name="mydl_framework",
    version="0.1.0",
    description="A simple deep learning framework",
    author="Your Name",
    packages=find_packages(exclude=["tests", ".venv", "data"]),
    install_requires=[
        "numpy",
        "pytest"
    ],
    python_requires=">=3.7",
)
