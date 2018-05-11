from os import path
from setuptools import setup

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='auto-sklearn-wrapper',
    version='0.0.6',
    packages=['auto_sklearn_wrapper'],
    url='https://github.com/Yatoom/auto-sklearn-auto_sklearn_wrapper',
    license='',
    author='Jeroen van Hoof',
    author_email='jeroen@jeroenvanhoof.nl',
    description='An Auto-Sklearn Wrapper that provides a compatibility layer between Auto-Sklearn and OpenML',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=["sklearn", "numpy"]
)
