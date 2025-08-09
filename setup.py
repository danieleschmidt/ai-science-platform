from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai_science_platform",
    version="0.1.0",
    description="AI-driven scientific discovery automation and research acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    url="https://github.com/danieleschmidt/ai-science-platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ai-science=ai_science_platform.cli:main",
        ],
    },
    keywords="ai, machine-learning, scientific-computing, research, discovery, automation",
    project_urls={
        "Documentation": "https://github.com/danieleschmidt/ai-science-platform/docs",
        "Source": "https://github.com/danieleschmidt/ai-science-platform",
        "Tracker": "https://github.com/danieleschmidt/ai-science-platform/issues",
    },
)
