"""Setup configuration for data-preproc package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="data-preproc",
    version="0.1.0",
    author="Benjamin Feuer",
    author_email="penfever@gmail.com",
    description="Data preprocessing for LLMs and VLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/penfever/data-preproc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "torch>=2.0.0",
        "fire>=0.5.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "addict>=2.4.0",
        "requests>=2.31.0",
        "rapidfuzz>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
        "vision": [
            "Pillow>=8.0.0",
            "av>=8.0.0",
            "librosa>=0.9.0",
        ],
        "toxicity": [
            "detoxify>=0.5.0",
            "open-clip-torch>=2.20.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-preproc=data_preproc.cli.preprocess:main",
        ],
    },
)