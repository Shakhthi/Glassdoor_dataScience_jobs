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


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
                                                training_pipeline_config.artifact_dir,
                                                training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_dir = os.path.join(
                                    self.data_validation_dir,
                                    training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_dir = os.path.join(
                                    self.data_validation_dir,
                                    training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path = os.path.join(
                                    self.valid_dir, 
                                    training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(
                                    self.valid_dir,
                                    training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path = os.path.join(
                                    self.invalid_dir,
                                    training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(
                                    self.invalid_dir,
                                    training_pipeline.TEST_FILE_NAME)
        self.drift_report_dir = os.path.join(
                                    self.data_validation_dir,
                                    training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                    training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        self.preprocessing_object_file_path = os.path.join(
                                    self.data_validation_dir,
                                    training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                    training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(
                                                training_pipeline_config.artifact_dir,
                                                training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path = os.path.join(
                                                self.data_transformation_dir,
                                                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_NAME.replace(".csv", ".npy"))
        
        self.transformed_test_file_path = os.path.join(
                                                self.data_transformation_dir,
                                                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                training_pipeline.DATA_TRANSFORMATION_TEST_FILE_NAME.replace(".csv", ".npy"))
        self.transformed_object_file_path = os.path.join(
                                                self.data_transformation_dir,
                                                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                training_pipeline.DATA_TRANSFORMATION_TRANSFORMER_OBJECT_FILE_NAME)
    
                                    
        
        

