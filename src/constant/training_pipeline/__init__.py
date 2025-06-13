import os
import sys

"""
Defining common constants for the training pipeline
"""

TARGET_COLUMN: str = "Result"
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

