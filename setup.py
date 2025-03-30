from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="polya_fields_visualization",
    version="0.1.0",
    author="Antipova Yulia",
    author_email="antipula38@gmail.com",
    description="Visualization of complex functions using Polya fields",
    url="https://github.com/antipula38/polya_fields_visualization.git",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'scipy>=1.7.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    keywords='complex-analysis visualization mathematics polya-fields',
    include_package_data=True
)
