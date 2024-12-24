import pandas as pd
import numpy as np
import os
import sys
import json
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransformation:
    """
    This class is responsible for preprocessing data by applying custom transformations, scaling, 
    and handling class imbalance using SMOTEENN.
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_file(filepath: str) -> pd.DataFrame:
        """Read a CSV file and return a DataFrame."""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """Create a data transformer pipeline based on schema configuration."""
        try:
            std_scaler = StandardScaler()
            min_max_scaler = MinMaxScaler()

            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('StandardScaler', std_scaler, num_features),
                    ('MinMaxScaler', min_max_scaler, mm_columns)
                ],
                remainder='passthrough'
            )

            return Pipeline(steps=[("Preprocessor", preprocessor)])
        except Exception as e:
            raise MyException(e, sys)

    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map 'Gender' column to binary values."""
        try:
            if 'Gender' in df.columns:
                df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical features."""
        try:
            return pd.get_dummies(df, drop_first=True)
        except Exception as e:
            raise MyException(e, sys)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename specific columns and ensure integer types for dummy columns."""
        try:
            column_mapping = {
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
            }
            df.rename(columns=column_mapping, inplace=True)

            for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            return df
        except Exception as e:
            raise MyException(e, sys)

    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the 'id' column if specified in the schema."""
        try:
            drop_column = self._schema_config['drop_columns']
            if drop_column in df.columns:
                df.drop(drop_column, axis=1, inplace=True)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Perform data transformation and return transformation artifacts."""
        try:
            if not self.data_validation_artifact.validation_status:
                raise MyException(self.data_validation_artifact.message, sys)

            train_df = self.read_file(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_file(self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Apply custom transformations
            for transform_func in [
                self._map_gender_column, self._drop_id_column,
                self._create_dummy_columns, self._rename_columns
            ]:
                input_feature_train_df = transform_func(input_feature_train_df)
                input_feature_test_df = transform_func(input_feature_test_df)

            preprocessor = self.get_data_transformer_object()

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys)
