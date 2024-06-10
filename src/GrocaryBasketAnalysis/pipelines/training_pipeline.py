import os
import sys
from src.GrocaryBasketAnalysis.logger import logging
from src.GrocaryBasketAnalysis.exception import customexception
from src.GrocaryBasketAnalysis.components.data_ingestion import data_ingestion
from src.GrocaryBasketAnalysis.components.data_transformation import data_transformation
from src.GrocaryBasketAnalysis.components.model_trainer import ModelTrainer




obj1=data_ingestion()
clean_list=obj1.initiate_data_ingestion()


obj2=data_transformation()
encoder_path,model_data=obj2.data_transform_initiated(clean_list)


obj3=ModelTrainer()
obj3.initate_model_training(model_data)
