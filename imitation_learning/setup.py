from setuptools import setup, find_packages

setup(
    name='low_altitude_nav',
    version='0.1.0',    
    url='url',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
)
