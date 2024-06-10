import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
from src.GrocaryBasketAnalysis.logger import logging
from src.GrocaryBasketAnalysis.exception import customexception
from src.GrocaryBasketAnalysis.utlis.utils import save_object

class dataingestionConfig:
    raw_data=os.path.join("artifacts","raw.csv")
    clean_data=os.path.join("artifacts","list.pkl")


class data_ingestion:
    def __init__(self):
        self.ingestion_config=dataingestionConfig()
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
           data=pd.read_csv(Path(os.path.join("notebooks/data","Assignment-1_Data.csv"))) 
           os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data)),exist_ok=True) 
           data.to_csv(self.ingestion_config.raw_data,index=False) 
           logging.info('row data saved ')  
           logging.info('remove null values ')
           data.dropna(subset=['Itemname'],inplace=True)
           logging.info('remove null values from item columns ')
           data.dropna(inplace=True)
           logging.info('remove null values from daatset ')
           data.drop_duplicates(inplace=True)
           logging.info('remove duplicate values from daatset ')
           d1=data.groupby('CustomerID')['Itemname'].apply(list)
           logging.info('create a list using all items WTR of customerid')
           save_object(obj=d1,file_path=self.ingestion_config.clean_data)
           logging.info('saved list data')
           return (self.ingestion_config.clean_data)
        except Exception as e:
           logging.info("exception during occured at data ingestion stage")
           raise customexception(e,sys)