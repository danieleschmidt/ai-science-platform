from setuptools import setup, find_packages

setup(
    name="ai_science_platform",
    version="0.1.0",
    description="AI-driven scientific discovery automation and research acceleration",
    author="Daniel Schmidt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies will be added based on research needs
    ],
    python_requires=">=3.8",
)
