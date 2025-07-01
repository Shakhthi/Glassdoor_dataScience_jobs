import os
import sys

import pandas as pd

from src.logging.logger import logging
from src.exception.exception_handler import ExceptionHandler

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

from src.utils.ml_utils.regression_metric import get_regression_score
from src.utils.ml_utils.estimator import DsEstimator
from src.utils.main_utils.utils import ( 
                load_object, save_object, load_yaml,
                load_numpy_array_data, evaluate_models
                )

from src.utils.ml_utils.regression_metric import RegressionMetricArtifact

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import mlflow
import dagshub

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Initializing Model Trainer Class")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ExceptionHandler(e, sys)
    
    def track_model(self, best_model, regressionmetric):
        try:
            logging.info("mlflow initiated.")
            with mlflow.start_run():
                r2_score = regressionmetric.r2_score
                mae = regressionmetric.mean_absolute_error
                mse = regressionmetric.mean_squared_error
                rmse = regressionmetric.root_mean_squared_error

                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("mean_absolute_score", mae)
                mlflow.log_metric("mean squared error", mse)
                mlflow.log_metric("root mean squared error", rmse)
                mlflow.sklearn.log_model(best_model, "model")
        
        except Exception as e:
            raise ExceptionHandler(e, sys)
    
    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            logging.info("Training the model")
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "RandomForestRegressor": RandomForestRegressor(verbose=1),
                "GradientBoostingRegressor": GradientBoostingRegressor(verbose=1),
                "SVR": SVR()
            }

            params = load_yaml(file_path = r"src\entity\ml_param_entity.yaml")
            logging.info(f"Parameters loaded: {params}")


            logging.info("Model and params are gathered.")
            model_report:dict=evaluate_models(
                                x_train=x_train, 
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test,  
                                models=models,
                                param=params)
            
            ## To get best model score from dict
            logging.info(f"Model report: {model_report}")
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            logging.info(f"Best model score: {best_model_score}")
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]


            y_train_pred=best_model.predict(x_train)
            logging.info(f"Training the model with best model name: {best_model}")
            regression_train_metric=get_regression_score(y_true=y_train, y_pred=y_train_pred)

            # tracking train metric with mlflow 
            logging.info("Tracking train metric with mlflow.")
            self.track_model(best_model, regression_train_metric)

            y_test_pred=best_model.predict(x_test)
            logging.info(f"Testing the model with best model name: {best_model}")
            regression_test_metric=get_regression_score(y_true=y_test, y_pred=y_test_pred)

            # tracking test metric with mlflow
            logging.info("Tracking test metric with mlflow.")
            self.track_model(best_model, regression_test_metric)

            logging.info("classification report ready.")
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            logging.info(f"Model directory path created: {model_dir_path}")

            Ds_jobs_Model=DsEstimator(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Ds_jobs_Model)
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")
            
            #model pusher
            save_object("final_model/model.pkl", best_model)
            logging.info("Model pusher is done.")
            

            ## Model Trainer Artifact
            model_trainer_artifact=ModelTrainerArtifact(
                                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                train_metric_artifact=regression_train_metric,
                                test_metric_artifact=regression_test_metric
                                )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ExceptionHandler(e, sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("data_transformation_artifact is being loaded in the model trainer class.")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            logging.info(f"Loading training data from: {train_file_path}")
            logging.info(f"Loading testing data from: {test_file_path}")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logging.info("train_model function is being called.")
            model_trainer_artifact=self.train_model(x_train, y_train, x_test, y_test)
            
            dagshub.init(repo_owner='Shakhthi', repo_name='ds_jobs', mlflow=True)
            return model_trainer_artifact
        except Exception as e:
            raise ExceptionHandler(e, sys)    

