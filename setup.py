from setuptools import setup, find_packages

setup(
    name='ibc',
    version='0.0.1',
    install_requires=[
        "imbalanced_learn == 0.9.0",
        "imblearn == 0.0",
        "numpy == 1.21.2",
        "pandas == 1.3.5",
        "scikit_learn == 1.2.1",
        "setuptools == 58.0.4",
        "torch == 1.10.2",
        "torchmetrics == 0.10.1"
    ],
    packages=find_packages(),
)
