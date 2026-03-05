from setuptools import setup, find_packages

setup(
    name="my_thesis_project",
    version="1.0.0",
    # find_packages() will look for directories with __init__.py
    packages=find_packages(),
    install_requires=[
    ],
    python_requires='>=3.7',
)