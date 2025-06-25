import os
import sys

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.logging.logger import logging
from src.exception.exception_handler import ExceptionHandler

from src.constant.training_pipeline import TARGET_COLUMN

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact


from src.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object
)

class DataTransformation:
    def __init__(self, 
                data_transformation_config: DataTransformationConfig,
                data_validation_artifact: DataValidationArtifact):
        """
        Initialize the DataTransformation class with configuration and validation artifacts.
        
        :param data_transformation_config: Configuration for data transformation.
        :param data_validation_artifact: Artifact containing validation results.
        """
        try:
            logging.info("Data Transformation Initialization started")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise ExceptionHandler(e, sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading the data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def get_feature_types(self, df: pd.DataFrame):
        try:
            logging.info("Identifying feature types in the DataFrame")
            feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
            num_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
            return num_cols, cat_cols
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def build_preprocessor(self, num_cols, cat_cols):
        try:
            logging.info("Building preprocessor for numerical and categorical features")
            num_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols)
            ])
            return preprocessor
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling missing values in the DataFrame")
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            return df
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            # Read validated train and test data
            logging.info("reading validated train and test data")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Remove duplicates
            logging.info("Removing duplicates from train and test data")
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            # Handle missing values
            logging.info("Handling missing values in train and test data")
            train_df = self.handle_missing_values(train_df)
            test_df = self.handle_missing_values(test_df)

            # Identify feature types
            logging.info("Identifying feature types in train and test data")
            num_cols, cat_cols = self.get_feature_types(train_df)

            # Build preprocessor
            logging.info("Building preprocessor for numerical and categorical features")
            preprocessor = self.build_preprocessor(num_cols, cat_cols)

            # Separate features and target
            logging.info("Separating features and target variable")
            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN]

            # Fit and transform
            logging.info("Fitting and transforming the data using preprocessor")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine features and target
            logging.info("Combining transformed features and target variable")
            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]

            # Save transformed arrays and preprocessor object
            logging.info("Saving transformed data and preprocessor object")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            # Create and return artifact
            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
        except Exception as e:
            raise ExceptionHandler(e, sys)


