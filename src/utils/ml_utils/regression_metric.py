import sys
from src.exception.exception_handler import ExceptionHandler
from src.logging.logger import logging

from src.entity.artifact_entity import RegressionMetricArtifact

from src.constant.training_pipeline import MODEL_TRAINER_EXPECTED_SCORE

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error


def get_regression_score(y_true, y_pred)->RegressionMetricArtifact:
    try:    
        r2_score_val = r2_score(y_true, y_pred)
        mean_absolute_error_score = mean_absolute_error(y_true, y_pred)
        mean_squared_error_score = mean_squared_error(y_true, y_pred)   
        root_mean_squared_error_score = root_mean_squared_error(y_true, y_pred)

        is_model_accepted = r2_score_val >= MODEL_TRAINER_EXPECTED_SCORE
        if not is_model_accepted:
            logging.info(f"Model is not accepted as r2_score: {r2_score_val} is less than expected score: {MODEL_TRAINER_EXPECTED_SCORE}")
        else:
            logging.info(f"Model is accepted as r2_score: {r2_score_val} is greater than or equal to expected score: {MODEL_TRAINER_EXPECTED_SCORE}")

        Regression_metric =  RegressionMetricArtifact(
                                    is_model_accepted = True,
                                    r2_score = r2_score_val,
                                    mean_absolute_error = mean_absolute_error_score,
                                    mean_squared_error = mean_squared_error_score,
                                    root_mean_squared_error = root_mean_squared_error_score,
                                    model_accuracy_file_path = None) # Placeholder, can be set later if needed    
                                        
        return Regression_metric
    except Exception as e:
        raise ExceptionHandler(e,sys)