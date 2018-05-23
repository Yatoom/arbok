from os import path
from setuptools import setup

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='arbok',
    version='0.1.4',
    packages=['arbok'],
    url='https://github.com/Yatoom/arbok',
    license='',
    author='Jeroen van Hoof',
    author_email='jeroen@jeroenvanhoof.nl',
    description='A wrapper toolbox that provides compatibility layers between TPOT and Auto-Sklearn and OpenML',
    long_description=long_description,
    install_requires=["sklearn", "auto-sklearn", "tpot", "numpy", "click", "Cython", "oslo.concurrency"],
    entry_points={
        'console_scripts': [
            'arbench = arbok.bench:cli',
        ],
    }
)
