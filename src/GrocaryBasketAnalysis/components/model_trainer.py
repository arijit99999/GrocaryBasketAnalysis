
import pandas as pd
import numpy as np
import os
import sys
from src.GrocaryBasketAnalysis.logger import logging
from src.GrocaryBasketAnalysis.exception import customexception
from src.GrocaryBasketAnalysis.utlis.utils import save_object
from src.GrocaryBasketAnalysis.utlis.utils import load_object


from mlxtend.frequent_patterns import apriori,association_rules


 
class ModelTrainerConfig:
    trained_model_file_path1= os.path.join('artifacts','support.csv')
    trained_model_file_path2= os.path.join('artifacts','confidence.csv')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,model_data):
        try:
            d2=pd.read_csv(model_data)
            logging.info('read model data')
            d2.drop('Unnamed: 0',axis=1,inplace=True)
            logging.info('remove column')
            support=apriori(d2,min_support=0.05,use_colnames=True)
            logging.info('support we calculate minimum thresold is 5%')
            print (support.sort_values(by='support',ascending=False).head())
            support.to_csv(self.model_trainer_config.trained_model_file_path1)
            logging.info('we saved support')
            confidence=association_rules(support,metric='confidence',min_threshold=.5)
            logging.info('confidence thresold is .5 minimum')
            print(confidence.sort_values(by='confidence',ascending=False).head())
            confidence.to_csv(self.model_trainer_config.trained_model_file_path2)
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)