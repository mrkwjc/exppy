import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="exppy",
    version="0.1.3",
    author="Marek Wojciechowski",
    author_email="mrkwjc@gmail.com",
    description="Numerical experiments in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrkwjc/exppy",
    packages=setuptools.find_packages(),
    keywords=['doe', 'numerical experiment'],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'pydoe2'],
    py_modules=['exppy']
)
