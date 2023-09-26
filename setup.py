from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="vectorgeo",
    version="0.1.0",
    description="Vector embeddings for the whole planet",
    author="Chris Krapu",
    author_email="ckrapu@gmail.com",
    url="https://github.com/ckrapu/vectorgeo",  # Replace with your project's repository URL
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
    ],
)
