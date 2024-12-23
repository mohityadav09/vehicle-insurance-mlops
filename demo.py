# from src.logger import logging
# from src.exception import MyException
# import sys

# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e

# -----------------------------------------------------------------
# from src import logger
# import logging

# logging.debug('yess it is working')

# ---------------------------------------------------------------------
## Runiing PROJ1 for fetching the records

# import pandas as pd
# from src.constants import COLLECTION_NAME,DATABASE_NAME
# from src.data_access.proj1_data import Proj1Data

# my_data = Proj1Data()

# data = my_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)

# print(len(data))

#----------------------------------------------------------------------------

## Training Pipeline
# import os
# from src.pipline.training_pipeline import TrainPipeline
# from src.components.data_ingestion import DataIngestion

# train_pipeline = TrainPipeline()
# train_pipeline.run_pipeline()

#----------------------------------------------------------------------------------------------
