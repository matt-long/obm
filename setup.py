#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from setuptools import setup
from os.path import exists

if exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

test_requirements = ['pytest']

setup(
    maintainer='Matthew Long',
    maintainer_email='mclong@ucar.edu',
    description='Ocean Box Models',
    install_requires=install_requires,
    license='Apache License 2.0',
    long_description=long_description,
    keywords='ocean biogeochemistry',
    name='obm',
    packages=['obm'],
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/matt-long/obm',
    zip_safe=False,
)
