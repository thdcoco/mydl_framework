# setup.py
from setuptools import setup, find_packages

setup(
    name="mydl_framework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "openai>=0.27.0",
        "pytest>=6.0"
    ],
)
