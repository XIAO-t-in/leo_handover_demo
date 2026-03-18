from setuptools import setup, find_packages

setup(
    name="leo_env",
    version="0.1.0",
    description="Low Earth Orbit satellite simulation environment (Starlink-inspired)",
    author="xiaoting",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
)
