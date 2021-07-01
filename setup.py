#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['rasterio>=1.2.0', 'numpy']

test_requirements = ['pytest>=3', ]

setup(
    author="Yann-Sebastien Tremblay-Johnston",
    author_email='yanns.tremblay@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Easily scale detected SAR float data to uint8 for easy viewing, and conversion to friendly file formats",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='seesar',
    name='seesar',
    packages=find_packages(include=['seesar', 'seesar.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/underchemist/seesar',
    version='0.1.0',
    zip_safe=False,
)
