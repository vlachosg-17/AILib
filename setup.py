from setuptools import setup, find_packages

setup(
    name='AILib', 
    version='0.1', 
    packages=find_packages(where="ailib") + find_packages(where="test") 
)