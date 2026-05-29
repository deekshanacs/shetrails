from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="forensic-ai-engine",
    version="1.0.0",
    author="Forensic AI Research Team",
    description="Research-grade multi-model image forensics and deepfake detection engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "PyWavelets>=1.4.1",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "piexif>=1.1.3",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "full": [
            "wandb>=0.16.0",
            "retinaface-pytorch>=0.0.8",
            "facenet-pytorch>=2.5.3",
            "albumentations>=1.3.1",
            "torchmetrics>=1.2.0",
            "grad-cam>=1.4.8",
            "av>=11.0.0",
            "transformers>=4.36.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "forensic-train=scripts.train_models:main",
            "forensic-analyze=scripts.run_inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)
