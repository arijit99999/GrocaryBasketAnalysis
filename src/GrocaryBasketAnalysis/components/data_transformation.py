
import os
import sys
import pandas as pd
from src.GrocaryBasketAnalysis.logger import logging
from src.GrocaryBasketAnalysis.exception import customexception
from src.GrocaryBasketAnalysis.utlis.utils import save_object
from src.GrocaryBasketAnalysis.utlis.utils import load_object
from mlxtend.preprocessing import TransactionEncoder

class datatransformationconfig:
    encoder_path=os.path.join("artifacts","encoder.pkl")
    mdoel_data=os.path.join("artifacts","model_data.csv")
class data_transformation:
    def __init__(self):
        self.prepocessor_path=datatransformationconfig()

        
    def data_transform_initiated(self,x):
       try:
          logging.info('trasnformation initiated')
          clean_list=load_object(x)
          logging.info('import list')
          pre=TransactionEncoder()
          res1=pre.fit_transform(clean_list)
          d2=pd.DataFrame(res1,columns=pre.columns_)
          d2.replace(False,0,inplace=True)
          d2.replace(True,1,inplace=True)
          d2.to_csv(self.prepocessor_path.mdoel_data)
          logging.info('save model data')
          save_object(obj=pre,file_path=self.prepocessor_path.encoder_path)
          logging.info('save encoder')
          return (self.prepocessor_path.encoder_path,self.prepocessor_path.mdoel_data)
       except Exception as e:
          logging.info("exception during occured at data tarnsformation initiation stage")
          raise customexception(e,sys)
