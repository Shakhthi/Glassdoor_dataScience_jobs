import os
import sys

import numpy as np
import pandas as pd

from src.logging.logger import logging
from src.exception.exception_handler import ExceptionHandler

from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
import mysql.connector
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

MONGODB_URL = os.getenv("MONGO_URI")
IP = os.getenv("MYSQL_HOST")
PORT = os.getenv("PORT")
PASSCODE = os.getenv("MYSQL_PASSCODE")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"Data Ingestion started with config: {data_ingestion_config}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ExceptionHandler(e, sys)
    
    def import_data_from_mongodb(self):
        try:
            logging.info("Importing data from MongoDB collection to dataframe")
            database_name = self.data_ingestion_config.mongodb_database_name
            collection_name = self.data_ingestion_config.mongodb_collection_name

            self.mongo_client = MongoClient(MONGODB_URL)
            collection = self.mongo_client[database_name][collection_name]
            train_data = pd.DataFrame(list(collection.find()))

            if "_id" in train_data.columns.to_list():
                train_data = train_data.drop(columns=["_id"], axis=1)
            train_data.replace({"na": np.nan}, inplace=True)

            logging.info("Importing data from MongoDB collection to dataframe completed successfully")
            return train_data
        except Exception as e:
            raise ExceptionHandler(e, sys)
        
    def import_data_from_mysql(self):
        try:
            logging.info("Importing data from MySQL database to dataframe")
            connection = mysql.connector.connect(
                host = IP,
                port = PORT,
                user = "root",
                password = PASSCODE,
                database = self.data_ingestion_config.mysql_database_name
            )

            query = f"SELECT * FROM {self.data_ingestion_config.mysql_table_name}"

            test_data = pd.read_sql(query, connection)
            test_data.replace({"na": np.nan}, inplace=True)
            logging.info("Importing data from MySQL database to dataframe completed successfully")
            return test_data
        
        except Exception as e:
            raise ExceptionHandler(e, sys)
        finally:
            if "connection" in locals() and connection.is_connected():
                connection.close()
                logging.info("MySQL connection closed")

    def split_data(self, data:pd.DataFrame, test_size: float = 0.2) -> tuple:
        """
        Splits the data into training and testing sets.
        
        :param data: DataFrame to split.
        :param test_size: Proportion of the dataset to include in the test split.
        :return: Tuple of training and testing DataFrames.
        """
        try:
            logging.info("Splitting data into training and testing sets") 

            if data.empty or data.empty:
                raise ValueError("Dataframes are empty. Please check the data source.")
            
            train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
            logging.info("Data split completed successfully")
            return train_data, test_data
        except Exception as e:
            raise ExceptionHandler(e, sys)
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion process")
            data = self.import_data_from_mongodb()

            data.to_csv(r"data\ds_jobs.csv", index=False, header=True)
            logging.info("Data imported from MongoDB and saved to CSV file")

            
            preprocessed_data = pd.read_csv(r"data\cleaned_ds_jobs.csv")
            train_data, test_data = self.split_data(preprocessed_data, test_size=self.data_ingestion_config.train_test_ratio)

            os.makedirs(self.data_ingestion_config.feature_store_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            train_file_path = self.data_ingestion_config.training_file_path
            test_file_path = self.data_ingestion_config.testing_file_path

            train_data.to_csv(train_file_path, index=False, header=True)
            test_data.to_csv(test_file_path, index=False, header=True)

            dataIngestionArtifact = DataIngestionArtifact(trained_file_path=train_file_path,
                                                          test_file_path=test_file_path)

            logging.info(f"Data ingestion completed successfully. Train file: {train_file_path}, Test file: {test_file_path}")
            return dataIngestionArtifact
        except Exception as e:
            raise ExceptionHandler(e, sys)
        


        
