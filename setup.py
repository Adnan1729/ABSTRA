from setuptools import setup, find_packages

setup(
    name="abstra",
    version="0.1.0",
    description="ABSTRA: Abstract Section-Targeted Reasoning Assessment",
    author="Adnan, Abbi, Gabe",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "captum==0.7.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "nltk>=3.8.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)