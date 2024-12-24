import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel


class ModelTrainer:
    """
    This is for training Model
    """

    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_train_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_train_config = model_train_config
        except Exception as e:
            raise MyException(e, sys)

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[RandomForestClassifier, ClassificationMetricArtifact]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info('Training Random Forest Classifier')
            
            # Splitting the dataset into features and target
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            logging.info('Data loaded successfully')

            # Initialize the model
            model = RandomForestClassifier(
                n_estimators=self.model_train_config.n_estimators,
                min_samples_split=self.model_train_config.min_samples_split,
                min_samples_leaf=self.model_train_config.min_samples_leaf,
                max_depth=self.model_train_config.max_depth,
                criterion=self.model_train_config.criterion,
                random_state=self.model_train_config.random_state
            )

            # Fit the model
            logging.info("Model training started...")
            model.fit(x_train, y_train)
            logging.info("Model training completed.")

            # Make predictions and calculate metrics
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')  # Added 'average' for multi-class
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            logging.info(f"Metrics calculated: Accuracy={accuracy}, F1={f1}, Precision={precision}, Recall={recall}")
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train and test data loaded")

            # Train model and get metrics
            model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            logging.info('Model and metrics successfully loaded')

            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded.")

            # Validate accuracy against threshold
            train_accuracy = accuracy_score(train_arr[:, -1], model.predict(train_arr[:, :-1]))
            if train_accuracy < self.model_train_config.expected_accuracy:
                logging.info("No model found with accuracy above the expected threshold")
                raise Exception("No model found with accuracy above the expected threshold")

            # Save the model
            logging.info("Saving new model as its performance is satisfactory.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=model)
            save_object(self.model_train_config.trained_model_file_path, my_model)
            logging.info("Saved final model object including preprocessing and trained model.")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_train_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
