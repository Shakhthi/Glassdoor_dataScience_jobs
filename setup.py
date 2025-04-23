from setuptools import setup, find_packages
from typing import List

from src.logging.logger import logging

def get_requirements(file_path:str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It removes any version specifiers and comments.
    """
    try:
        logging.info(f"Reading requirements from {file_path}")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and line != "-e .":
                requirements.append(line)
        
        return requirements
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No requirements will be installed.")
        return []

setup(
    name = "Glassdoor DataScience Jobs",
    version = "0.0.1",
    author = "MK",
    author_email = "sakthikaliappan7797@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt"),
)


