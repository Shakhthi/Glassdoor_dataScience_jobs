import sys
import os

from src.exception.exception_handler import ExceptionHandler

import pandas as pd

import yaml

def load_yaml(file_path) -> dict:
    """
    Load a YAML file and return its content.
    
    :param file_path: Path to the YAML file.
    :return: Content of the YAML file as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ExceptionHandler(f"Error loading YAML file {file_path}: {e}")
    
def save_yaml(data, file_path) -> None:
    """
    Save data to a YAML file.
    
    :param data: Data to be saved.
    :param file_path: Path where the YAML file will be saved.
    """
    try:
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        raise ExceptionHandler(f"Error saving YAML file {file_path}: {e}")
    
def write_to_yaml(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to a YAML file. If the file exists and replace is False, append to the file.
    
    :param file_path: Path to the YAML file.
    :param content: Content to write to the YAML file.
    :param replace: If True, replace the existing file; if False, append to it.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ExceptionHandler(e, sys)