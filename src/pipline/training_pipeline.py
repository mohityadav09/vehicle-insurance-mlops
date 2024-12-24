import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig)

from src.entity.artifact_entity import (DataIngestionArtifact,
                                        DataValidationArtifact,
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact)

class TrainPipeline:
    """
    TrainPipeline class orchestrates the steps required to build and train a machine learning model pipeline.
    It handles data ingestion, validation, transformation, and model training.
    """
    def __init__(self):
        """
        Initializes configurations for all the pipeline stages.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_train_config = ModelTrainerConfig()
        except Exception as e:
            raise MyException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process.

        :return: DataIngestionArtifact containing details of the ingested data.
        """
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info('Data ingestion completed successfully.')
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Initiates the data validation process.

        :param data_ingestion_artifact: Artifact from the data ingestion stage.
        :return: DataValidationArtifact containing validation details.
        """
        try:
            logging.info('Starting data validation.')
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info('Data validation completed successfully.')
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_transformation(self, 
                                   data_ingestion_artifact: DataIngestionArtifact,
                                   data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Initiates the data transformation process.

        :param data_ingestion_artifact: Artifact from the data ingestion stage.
        :param data_validation_artifact: Artifact from the data validation stage.
        :return: DataTransformationArtifact containing transformation details.
        """
        try:
            logging.info('Starting data transformation.')
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info('Data transformation completed successfully.')
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Initiates the model training process.

        :param data_transformation_artifact: Artifact from the data transformation stage.
        :return: ModelTrainerArtifact containing details of the trained model.
        """
        try:
            logging.info('Starting model training.')
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_train_config=self.model_train_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info('Model training completed successfully.')
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self):
        """
        Orchestrates the entire pipeline by running each stage in sequence.
        """
        try:
            logging.info('Pipeline execution started.')
            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact)

            # Model Training
            self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            logging.info('Pipeline execution completed successfully.')
        except Exception as e:
            raise MyException(e, sys)
