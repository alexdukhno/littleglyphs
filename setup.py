import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="littleglyphs",

    description="A library for investigating the performance of image classifiers on procedurally generated letter-like images",

    author="Alex Dukhno",

    packages=find_packages(
        exclude=['notebooks','output','scripts','data','figures',
        '__pycache__,','.git','.ipynb_checkpoints']
    ),

    long_description=read('README.md'),
)
