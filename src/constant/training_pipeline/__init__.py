import os
import sys

"""
Defining common constants for the training pipeline
"""

TARGET_COLUMN: str = "salary_avg_estimate"
PIPELINE_NAME: str = "Ds_jobs_pipeline"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "Glassdoor_job_postings.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")
SAVED_MODEL_DIR: str = os.path.join("saved_models")
MODEL_FILE_NAME: str = "model.pkl"

""""
Data ingestion constants starts with DATA_INGESTION var name
"""

DATA_INGESTION_COLLECTION_NAME: str = "ds_jobs"
DATA_INGESTION_DATABASE_NAME: str = "MK"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_RATIO =  0.2

DATA_INGESTION_MYSQL_DATABASE_NAME: str = "ds_jobs"
DATA_INGESTION_MYSQL_TABLE_NAME: str = "jobs"


"""
Data Validation constants starts with DATA_VALIDATION var name
"""
DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "valid_data"
DATA_VALIDATION_INVALID_DIR:str = "invalid_data"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME:str = "preprocessing.pkl"

"""
Data Transformation constants starts with DATA_TRANSFORMATION var name
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train_transformed.csv"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test_transformed.csv"

DATA_TRANSFORMATION_TRANSFORMER_OBJECT_FILE_NAME: str = "transformer.pkl"

"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "final_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

TRAINING_BUCKET_NAME = "ds_jobs"

AWS_BUCKET_NAME: str = "ds-jobs-aws-bucket"




