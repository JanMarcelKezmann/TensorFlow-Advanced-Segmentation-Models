import io
import os
import sys
import setuptools

# Package meta-data.
NAME = 'tensorflow_advanced_segmentation_models'
DESCRIPTION = 'A Python Library for High-Level Semantic Segmentation Models based on TensorFlow and Keras.'
URL = 'https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models'
EMAIL = 'jankezmann@t-online.de'
AUTHOR = 'Jan Marcel Kezmann'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

here = os.path.abspath(os.path.dirname(__file__))

try:
    if sys.platform == 'darwin':
        with open(os.path.join(here, 'requirements_macos.txt'), encoding='utf-8') as f:
            REQUIRED = f.read().split('\n')
    else:
        with open(os.path.join(here, 'requirements_windows.txt'), encoding='utf-8') as f:
            REQUIRED = f.read().split('\n')
except:
    REQUIRED = []
    
# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
   with open(os.path.join(here, NAME, '__version__.py')) as f:
       exec(f.read(), about)
else:
    about['__version__'] = VERSION


# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=about["__version__"],
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(exclude=("images", "examples")),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=REQUIRES_PYTHON,
)
