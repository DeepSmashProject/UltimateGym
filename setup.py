from setuptools import setup
from glob import glob
from os.path import basename
from os.path import splitext

with open('README.rst') as f:
    readme = f.read()

with open('requirements.txt') as f:
    all_reqs = f.read().split('\n')
install_requires = [x.strip() for x in all_reqs]

setup(
    name="ultimate_gym",
    version="0.0.0",
    packages=['ultimate_gym'],
    install_requires = install_requires,
    description='',
    long_description=readme,
    author='DeepSmash',
    author_email='deepsmash0@gmail.com',
    license='MIT',
)