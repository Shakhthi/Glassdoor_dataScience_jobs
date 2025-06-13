import os
import sys

from datetime import datetime

from src.constant import training_pipeline 

class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        timestamp = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp = timestamp 

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_ingestion_dir = os.path.join(
                                                training_pipeline_config.artifact_dir,
                                                training_pipeline.DATA_INGESTION_DIR_NAME)
        
        self.feature_store_dir = os.path.join(
                                                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE)
        
        self.training_file_path = os.path.join(
                                                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                                training_pipeline.TRAIN_FILE_NAME)
        
        self.testing_file_path = os.path.join(
                                                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                                training_pipeline.TEST_FILE_NAME)
        
        self.train_test_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_RATIO
        self.mongodb_database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.mongodb_collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        self.mysql_database_name = training_pipeline.DATA_INGESTION_MYSQL_DATABASE_NAME
        self.mysql_table_name = training_pipeline.DATA_INGESTION_MYSQL_TABLE_NAME
        

