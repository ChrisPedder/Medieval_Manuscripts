from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A project to retrain the VGG16 network with a classifier to spot a particular author of medieval manuscripts.',
    author='Chris Pedder',
    license='MIT',
)
