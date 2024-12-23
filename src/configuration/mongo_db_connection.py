import pymongo
import certifi
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()



from src.constants import DATABASE_NAME, MONGODB_URL_KEY
from src.logger import logging
from src.exception import MyException

# Get the CA certificate for secure connection
ca = certifi.where()

class MongoDBClient:
    """
    This class is responsible for connecting to the MongoDB database.
    """
    client = None  # Class-level client to ensure a single instance

    def __init__(self, database_name: str = DATABASE_NAME):
        try:
            # Fetch MongoDB URL from environment
            mongodb_url = os.getenv("MONGODB_URL_KEY")
            if not mongodb_url:
                raise ValueError(f"Environment variable  is not set.")
            
            # Initialize the MongoDB client only once
            if MongoDBClient.client is None:
                MongoDBClient.client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)
                logging.info("MongoDB client created successfully.")
            
            # Use the shared client instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            logging.info(f"Connected to MongoDB database: {database_name}")
        
        except Exception as e:
            raise MyException(e, sys)
