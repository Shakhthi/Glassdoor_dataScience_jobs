import sys
import os

from src.exception.exception_handler import ExceptionHandler
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from src.constant.training_pipeline import AWS_BUCKET_NAME
from src.cloud.s3_syncer import S3Sync

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    # data gathered from source(mongodb)
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise ExceptionHandler(e,sys)
        
    # checking for data mismatches in data
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            logging.info("Initiated the data Validation")
            data_validation_artifact=data_validation.validate_data()
            return data_validation_artifact
        except Exception as e:
            raise ExceptionHandler(e, sys)

    # data transformed for model training
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            logging.info("transformer component initiated.")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise ExceptionHandler(e,sys)

    # train and find the best model    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            logging.info("model trainer component initiated.")
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            raise ExceptionHandler(e, sys)
    def create_bucket(self, bucket_name:str):
        try:
            logging.info(f"Creating bucket: {bucket_name}")
            self.s3_sync.make_bucket(bucket_name=bucket_name)
        except Exception as e:
            raise ExceptionHandler(e, sys)
        
    def sync_artifact_dir_to_s3(self):
        try:
            logging.info("artifacts syncing to s3 initiated.")

            aws_bucket_url = f"s3://{AWS_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir, aws_bucket_url = aws_bucket_url)
            logging.info("Artifacts are pushed to s3 bucket.")
        except Exception as e:
            raise ExceptionHandler(e, sys)
        
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{AWS_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir, aws_bucket_url = aws_bucket_url)
            logging.info("saved model pushed to s3 bucket.")
        except Exception as e:
            raise ExceptionHandler(e,sys)

    # driver function for this class    
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            self.create_bucket(AWS_BUCKET_NAME)
            self.sync_artifact_dir_to_s3()
            # self.sync_saved_model_dir_to_s3()
            
            return model_trainer_artifact
        except Exception as e:
            raise ExceptionHandler(e,sys)