"""
Setup script for GraphFC package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graphfc",
    version="0.1.0",
    author="GraphFC Contributors",
    author_email="",
    description="A Graph-based Verification Framework for Fact-Checking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/graphfc",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "graphfc-demo=examples.interactive_demo:main",
            "graphfc-eval=examples.run_evaluation:main",
            "graphfc-ablation=examples.run_ablation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
    keywords=[
        "fact-checking",
        "graph-based",
        "natural-language-processing",
        "large-language-models",
        "knowledge-graphs",
        "verification",
        "misinformation-detection"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/graphfc/issues",
        "Source": "https://github.com/your-username/graphfc",
        "Documentation": "https://github.com/your-username/graphfc/README.md",
    },
)