from setuptools import setup

import sys

if sys.version_info < (3,1):
    sys.exit("Only Python 3.1 and greater is supported") 

setup(
    name='bbstrader',
    version='1.0',
    packages=['bbstrader'],
    url='https://github.com/bbalouki/BBSTrade',
    license='The MIT License (MIT)',
    author='Bertin Balouki SIMYELI',
    author_email='bbalouki@outlook.com',
    description='Simplified Investment & Trading Toolkit'
)
