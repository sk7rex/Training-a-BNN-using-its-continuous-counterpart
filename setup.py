from setuptools import setup, find_packages

setup(
    name="contbnn",
    version="0.1.0",
    author="Nikita Izmailov",
    author_email="nikitaizmaylovv@yandex.ru",
    description="A package for training a BNN using its continuous counterpart. This package is a part of a final qualifying work.",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "matplotlib"
    ],
    python_requires=">=3.12",
)