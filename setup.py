from setuptools import find_packages, setup

setup(
    name='ama-learn',
    packages=find_packages(include=['ama-learn']),
    version='0.1.0',
    url='https://github.com/Amayas29/ama-learn',
    description='Implementation of machine learning algorithms and models in python',
    author='Amayas29',
    license='MIT',
    install_requires=["numpy"]
)
