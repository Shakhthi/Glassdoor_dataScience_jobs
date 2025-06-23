import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

from src.constant import training_pipeline
from src.entity.config_entity import (TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig)
from src.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact)

from src.logging.logger import logging
from src.exception.exception_handler import ExceptionHandler

if __name__ == '__main__':
    try:
        # Initialize training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating data ingestion")
        data_ingestion_artifact: DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")

        # Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiating data validation")
        data_validation_artifact: DataValidationArtifact = data_validation.validate_data()
        logging.info("Data validation completed successfully")

    except Exception as e:
        raise ExceptionHandler(e, sys)  # Use ExceptionHandler to handle exceptions
        raise e