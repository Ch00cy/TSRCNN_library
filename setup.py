import setuptools

with open("README", "r", encoding='UTF8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="TSRCNN", # Replace with your own PyPI username(id)
    version="0.0.12",
    author="Yeonhee Choo",
    author_email="chooyeonhee99@gmail.com",
    description="TSRCNN_Texture Synthesis with Rotation using CNN package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ch00cy/TSRCNN_library",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8'
)