from setuptools import setup, find_packages

setup(
    name="digital-arrest-scam-intelligence",
    version="0.1.0",
    description="AI-Based Digital Arrest Scam Intelligence & Early Threat Detection System",
    author="Aditya",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "torch-geometric>=2.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "librosa>=0.10.0",
        "networkx>=3.1",
        "pyyaml>=6.0.0",
        "loguru>=0.7.0",
    ],
)
