import os
import sys
import pandas as pd
import numpy as np

from src.constants import COLLECTION_NAME
from src.configuration.mongo_db_connection import MongoDBClient
from src.logger import logging
from src.exception import MyException


class Proj1Data:
    """
    This class is responsible for accessing data from a MongoDB database
    and exporting it as a Pandas DataFrame.
    """

    def __init__(self):
        try:
            # Initialize MongoDB client
            self.mongo_client = MongoDBClient()
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: str = None) -> pd.DataFrame:
        """
        Fetches a MongoDB collection and exports it as a Pandas DataFrame.

        :param collection_name: Name of the MongoDB collection.
        :param database_name: Name of the MongoDB database. Defaults to the one connected by MongoDBClient.
        :return: DataFrame containing the collection data.
        """
        try:
            # Access the collection
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                db = self.mongo_client.client[database_name]
                collection = db[collection_name]

            # Fetch all data from the collection
            data = list(collection.find())
            logging.info(f"Fetched {len(data)} records from the collection '{collection_name}'.")

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Handle the "_id" field
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
                logging.info("Dropped '_id' column from DataFrame.")

            # Replace placeholder values with NaN
            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"Replaced placeholder values with NaN in DataFrame.")

            return df

        except Exception as e:
            raise MyException(e, sys)
