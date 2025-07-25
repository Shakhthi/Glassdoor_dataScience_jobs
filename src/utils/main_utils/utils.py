import sys
import os

from src.exception.exception_handler import ExceptionHandler
from src.logging.logger import logging

import numpy as np
import pandas as pd

import yaml
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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
            yaml.safe_dump(content, file)
    except Exception as e:
        raise ExceptionHandler(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ExceptionHandler(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ExceptionHandler(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise ExceptionHandler(e, sys) from e
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise ExceptionHandler(e, sys) from e
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        for i, model_name in enumerate(models):
            model = models[model_name]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3, scoring='r2', n_jobs=-1)
            gs.fit(x_train, y_train)
            best_params = gs.best_params_

            model.set_params(**best_params)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            # Overfitting gap
            overfit_gap = train_model_score - test_model_score
            report[model_name] = {
                'best_params': best_params,
                'train_score': train_model_score,
                'test_score': test_model_score,
                'overfit_gap': overfit_gap,
                'cv_score': gs.best_score_
            }
        return report
    except Exception as e:
        raise ExceptionHandler(e, sys)