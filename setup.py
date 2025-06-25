from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-theorem-prover",
    version="0.1.0",
    author="Tu Nombre",
    author_email="tu.email@example.com",
    description="Generador automático de teoremas usando ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuusuario/AI-Theorem-Prover",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0",
        "sympy>=1.10.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "web": [
            "flask>=2.0.0",
            "fastapi>=0.75.0",
            "uvicorn>=0.17.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "theorem-generator=theorem_generator:main",
        ],
    },
)