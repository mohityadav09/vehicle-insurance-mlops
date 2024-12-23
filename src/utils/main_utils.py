import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging

def read_yaml(file_path:str) ->dict:
    """
    this function read Yaml file from its file path and load into dictionary
    """ 
    try:
        with open(file_path,'rb') as file:
            content = yaml.safe_load(file)

        logging.info('Yaml file loaded successfully')    

        return content
    except Exception as e:
        raise MyException(e,sys)
    
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    This function use to write yamal file at the given file path 
    inputs:
    file_path : Location where file is going to save
    content : Object type that we want to save in yaml file
    replace: True/False we want to replace the following content if it exists
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            yaml.dump(content,file)

    except Exception as e:
        raise MyException(e,sys)
    





def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise MyException(e, sys) from e
    


    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MyException(e, sys) from e
    



def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e
    



def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise MyException(e, sys) from e
