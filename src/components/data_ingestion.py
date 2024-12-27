import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    """"spliiting data into train test spilit"""
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise MyException(e,sys)


    def export_data_into_feature_store(self)->DataFrame:
        try:
            my_data = Proj1Data()

            dataframe = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)

            logging.info('Data imported successfully')

            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            if 'id' in dataframe.columns:
                dataframe.drop('id',axis=1,inplace=True)
            return dataframe
        
        except Exception as e:
            raise MyException(e,sys)
        
    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        try:
            test_size = self.data_ingestion_config.train_test_split_ratio
            train_set,test_set = train_test_split(dataframe,test_size=test_size,random_state=22,shuffle=True)
            logging.info("Performed train test split on the dataframe")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise MyException(e,sys)

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
         
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            data = self.export_data_into_feature_store()
            self.split_data_as_train_test(data)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e








            