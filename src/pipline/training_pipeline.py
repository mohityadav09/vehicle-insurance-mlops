import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig,
                                      ModelPusherConfig)

from src.entity.artifact_entity import (DataIngestionArtifact,
                                        DataValidationArtifact,
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact,
                                        ModelEvaluationArtifact,
                                        ModelPusherArtifact)

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
            self.model_eval_config = ModelEvaluationConfig()
            self.model_pusher_config = ModelPusherConfig()


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

    def start_model_evaluation(self,data_ingestion_artifact:DataIngestionArtifact,
                               data_transformation_artifact:DataTransformationArtifact,
                               model_trainer_artifact:ModelTrainerArtifact,
                               model_eval_config:ModelEvaluationConfig)-> ModelEvaluationArtifact:
        try:
            logging.info('start model evaluation')
            model_eval = ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,
                                         model_trainer_artifact=model_trainer_artifact,
                                         model_eval_config=model_eval_config,
                                         data_transformaton_artifact=data_transformation_artifact)
            result = model_eval.initiate_model_evaluation()
            logging.info('Evaluation done')
            return result
        except Exception as e:
            raise MyException(e,sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.model_pusher_config
                                       )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys)
    
                

    def run_pipeline(self)-> None:
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
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            # Model Evaluation

            model_eval_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                              data_transformation_artifact=data_transformation_artifact,
                                                              model_trainer_artifact=model_trainer_artifact,
                                                              model_eval_config=self.model_eval_config)
            
            if model_eval_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_eval_artifact)

            else:
                print('Model is not accepted')


            logging.info('Pipeline execution completed successfully.')

            return None
        except Exception as e:
            raise MyException(e, sys)
