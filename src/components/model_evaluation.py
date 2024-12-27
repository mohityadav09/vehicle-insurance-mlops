from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact,DataTransformationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pickle
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """
    This class is used to evaluate our trained model and compare the performance of the previously deployed model
    and trained model
    """

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,model_trainer_artifact:ModelTrainerArtifact,
                 model_eval_config: ModelEvaluationConfig,data_transformaton_artifact:DataTransformationArtifact)->None:
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformaton_artifact

        except Exception as e:
            raise MyException(e,sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)

    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df


    def evaluate_model(self)->EvaluateModelResponse:
        """
        This method loads test data from data_ingestion_artifact , do some basic preprocessing
        apply preprocessing pipeline from data_transformation_artifact , compare the f1 score of the model
        from s3 storage and already calculated f1_score in model_trainer_artifact
        """ 
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x,y = test_df.drop(TARGET_COLUMN,axis =1),test_df[TARGET_COLUMN]

            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            logging.info("Test data loaded and now transforming it for prediction...")

            preprocessing = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            logging.info(f'preprocessing object loaded from filepath {self.data_transformation_artifact.transformed_object_file_path}')

            x = preprocessing.transform(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)

            logging.info('train model loaded')

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")

            temp_best_model_f1_score = 0 if best_model_f1_score is None else best_model_f1_score

            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score= temp_best_model_f1_score,
                                           is_model_accepted= True if trained_model_f1_score > temp_best_model_f1_score else False,
                                           difference= trained_model_f1_score-  temp_best_model_f1_score)

            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)
    
    def initiate_model_evaluation(self)-> ModelEvaluationArtifact:
        """
        Initiate the model evaluation and return the result into ModelEvaluationArtifact
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e





