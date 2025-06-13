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
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion process")
            train_data = self.import_data_from_mongodb()
            test_data = self.import_data_from_mysql()

            if train_data.empty or test_data.empty:
                raise ValueError("Dataframes are empty. Please check the data source.")

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
        
if __name__ == "__main__":
    try:
        
        trainconfig = TrainingPipelineConfig()
        print(f"{trainconfig.timestamp} - Data Ingestion started")
        data_ingestion = DataIngestion(data_ingestion_config = DataIngestionConfig(trainconfig))
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise ExceptionHandler(e, sys)

        
