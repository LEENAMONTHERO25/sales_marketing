from src.Sales_Marketing.components.data_ingestion import DataIngestion

from src.Sales_Marketing.components.data_transformation import DataTransformation

from src.Sales_Marketing.components.model_trainer import ModelTrainer



import os
import sys
from src.Sales_Marketing.logger import logging
from src.Sales_Marketing.exception import customexception
import pandas as pd

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_trainer(train_arr,test_arr)