import os
import sys
import pickle
from src.GrocaryBasketAnalysis.logger import logging
from src.GrocaryBasketAnalysis.exception import customexception



def save_object(file_path, obj):
    try:
         dir_path = os.path.dirname(file_path)
         os.makedirs(dir_path, exist_ok=True)
         with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
         logging.info('pickle dump succesfully')
    except Exception as e:
        logging.info('error got in  pickle dump')
        raise customexception(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        logging.info('pickle loaded succesfully')
    except Exception as e:
        logging.info('error got in  pickle load')
        raise customexception(e, sys)