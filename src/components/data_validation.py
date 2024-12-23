from src.entity.artifact_entity import DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils import main_utils
from src.constants import SCHEMA_FILE_PATH
import os
import json
import sys
import pandas as pd
import numpy as np
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact):

        try:
            self.data_validation_config = DataValidationConfig()
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = main_utils.read_yaml(SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e,sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """ Validate the columns from  train data (loaded from database) and the data we need for the
        actual taraining
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self.schema_config['numerical_column']:
                if column not in df.columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")        

            for column in self.schema_config['categorical_column']:
                if column not in df.columns:
                    missing_categorical_columns.append(column) 

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")


            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise MyException(e, sys)


    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self):
        """
        Method: to initiate the DataVlidation
        """ 
        try:
            err = ""
            logging.info('Initiate data Validation')
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df =  self.read_data(self.data_ingestion_artifact.test_file_path)
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All required columns present in training dataframe: {status}")

            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe. "
            else:
                logging.info(f"All required columns present in testing dataframe: {status}")

            # Validating col dtype for train/test df
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All categorical/int columns present in training dataframe: {status}")

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logging.info(f"All categorical/int columns present in testing dataframe: {status}")

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path,'w') as file:
                json.dump(validation_report,file,indent=4)  

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise MyException(e, sys)  






