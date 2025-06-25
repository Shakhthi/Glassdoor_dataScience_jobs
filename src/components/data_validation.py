import sys
import os
import yaml
import pandas as pd
from scipy.stats import ks_2samp
from src.logging.logger import logging
from src.exception.exception_handler import ExceptionHandler
from src.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils.utils import load_yaml

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            logging.info("Initiated Automated Data Validation Class.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_file = load_yaml(SCHEMA_FILE_PATH)
            self.fix_report = {"train": {}, "test": {}}
        except Exception as e:
            raise ExceptionHandler(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file:{file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def fix_number_of_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = list(self._schema_file['columns'].keys())
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns]
        for col in missing_cols:
            df[col] = pd.NA
        df = df[expected_columns]  # reorder and drop extras
        return df

    def fix_column_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = self._schema_file['columns']
        for col, dtype in expected_columns.items():
            if col in df.columns:
                try:
                    if dtype == 'string':
                        df[col] = df[col].astype(str)
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception:
                    df[col] = pd.NA
        return df

    def fix_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna("missing", inplace=True)
        return df

    def fix_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def fix_data_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        # Not much can be done automatically, so just log
        return df

    def validate_and_fix(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        # Number of columns
        if not self.validate_number_of_columns(df):
            self.fix_report[split]['number_of_columns'] = 'fixed'
            df = self.fix_number_of_columns(df)
        else:
            self.fix_report[split]['number_of_columns'] = 'ok'
        # Column datatypes
        if not self.validate_column_datatypes(df):
            self.fix_report[split]['column_datatypes'] = 'fixed'
            df = self.fix_column_datatypes(df)
        else:
            self.fix_report[split]['column_datatypes'] = 'ok'
        # Missing values
        if not self.validate_missing_values(df):
            self.fix_report[split]['missing_values'] = 'fixed'
            df = self.fix_missing_values(df)
        else:
            self.fix_report[split]['missing_values'] = 'ok'
        # Duplicates
        if not self.validate_duplicate_rows(df):
            self.fix_report[split]['duplicate_rows'] = 'fixed'
            df = self.fix_duplicate_rows(df)
        else:
            self.fix_report[split]['duplicate_rows'] = 'ok'
        # Data distribution (no auto-fix)
        if not self.validate_data_distribution(df):
            self.fix_report[split]['data_distribution'] = 'not_fixed'
        else:
            self.fix_report[split]['data_distribution'] = 'ok'
        return df

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            expected_columns = self._schema_file['columns']
            expected_columns = list(expected_columns.keys())
            actual_columns = df.columns.tolist()
            return set(expected_columns) == set(actual_columns)
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def validate_column_datatypes(self, df: pd.DataFrame) -> bool:
        try:
            expected_columns = self._schema_file['columns']
            for col, dtype in expected_columns.items():
                if col in df.columns:
                    if dtype == 'string':
                        expected_dtype = 'object'
                    else:
                        expected_dtype = dtype
                    if str(df[col].dtype) != expected_dtype:
                        return False
            return True
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        try:
            return not df.isnull().any().any()
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def validate_duplicate_rows(self, df: pd.DataFrame) -> bool:
        try:
            return df.duplicated().sum() == 0
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def validate_data_distribution(self, df: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_file['numerical_columns']
            for col in numerical_columns:
                if col in df.columns:
                    ks_statistic, p_value = ks_2samp(df[col], df[col].dropna())
                    if p_value < 0.05:
                        return False
            return True
        except Exception as e:
            raise ExceptionHandler(e, sys)

    def validate_data(self) -> DataValidationArtifact:
        try:
            logging.info("Starting automated data validation and fixing process.")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            train_df = self.validate_and_fix(train_df, 'train')
            test_df = self.validate_and_fix(test_df, 'test')

            # Save fixed data
            is_valid = all(v == 'ok' or v == "fixed" for split in self.fix_report.values() for v in split.values())
            if is_valid:
                # Save to valid paths from config
                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
            else:
                # Save original (broken) data to invalid paths from config
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_test_file_path), exist_ok=True)
                self.read_data(self.data_ingestion_artifact.trained_file_path).to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                self.read_data(self.data_ingestion_artifact.test_file_path).to_csv(self.data_validation_config.invalid_test_file_path, index=False)


                # Save fixed data to valid paths from config
                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            # Write YAML report
            report_path = os.path.join(self.data_validation_config.data_validation_dir, 'data_validation_report.yaml')
            with open(report_path, 'w') as f:
                yaml.dump(self.fix_report, f)
            logging.info(f"Automated data validation and fixing report written to {report_path}")

             # Log all DataValidationArtifact paths before returning
            logging.info(f"DataValidationArtifact paths:")
            logging.info(f"  valid_train_file_path: {self.data_validation_config.valid_train_file_path if is_valid else None}")
            logging.info(f"  valid_test_file_path: {self.data_validation_config.valid_test_file_path if is_valid else None}")
            logging.info(f"  invalid_train_file_path: {self.data_validation_config.invalid_train_file_path if not is_valid else None}")
            logging.info(f"  invalid_test_file_path: {self.data_validation_config.invalid_test_file_path if not is_valid else None}")
            logging.info(f"  drift_report_file_path: {report_path}")

            return DataValidationArtifact(
                validation_status = is_valid,
                valid_train_file_path = self.data_validation_config.valid_train_file_path if is_valid else None,
                valid_test_file_path = self.data_validation_config.valid_test_file_path if is_valid else None,
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path if not is_valid else None,
                invalid_test_file_path = self.data_validation_config.invalid_test_file_path if not is_valid else None,
                drift_report_file_path = report_path
            )
        except Exception as e:
            raise ExceptionHandler(e, sys)
