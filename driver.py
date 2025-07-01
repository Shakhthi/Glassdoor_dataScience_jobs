import sys
import warnings

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


from src.entity.config_entity import (
    TrainingPipelineConfig, 
    DataIngestionConfig, 
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig)

from src.entity.artifact_entity import (
    DataIngestionArtifact, 
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact)

from src.logging.logger import logging
from src.exception.exception_handler import ExceptionHandler

warnings.filterwarnings("ignore")


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
        Data_Validation_Artifact: DataValidationArtifact = data_validation.validate_data()
        logging.info("Data validation completed successfully")

        # Data Transformation
        data_validation_artifact = Data_Validation_Artifact
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                  data_transformation_config=DataTransformationConfig(training_pipeline_config))
        logging.info("Initiating data transformation")
        data_transformation_artifact:DataTransformationArtifact = data_transformation.initiate_data_transformation()

        # Model Trainer
        logging.info("Initiating model training")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact: ModelTrainerArtifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed successfully")

        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
    except Exception as e:
        raise ExceptionHandler(e, sys)  # Use ExceptionHandler to handle exceptions
    