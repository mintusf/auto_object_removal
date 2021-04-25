from setuptools import setup, find_packages

setup(
    name='auto-object-removal',
    version='0.0.0',
    author='fmintus',
    description='auto-object-removal',
    packages = find_packages(exclude=("tests*")),
)