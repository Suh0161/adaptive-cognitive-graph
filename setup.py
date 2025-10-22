"""
Setup script for ACG package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

setup(
    name="acg",
    version="1.1.0",
    author="afif amir",
    description="Adaptive Cognitive Graph: A scalable architecture for reasoning-centric language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/acg",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "distributed": [
            "deepspeed>=0.12.0",
            "transformers>=4.35.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "tensorboard>=2.14.0",
            "wandb>=0.15.0",
        ],
    },
    keywords="machine-learning deep-learning transformer mixture-of-experts language-model pytorch",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/acg/issues",
        "Source": "https://github.com/yourusername/acg",
        "Documentation": "https://github.com/yourusername/acg#readme",
    },
)
