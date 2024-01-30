
import pandas as pd
import numpy as np
from src.Sales_Marketing.logger import logging
from src.Sales_Marketing.exception import customexception


import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def create_outlet_years(self, data):
        # Convert "Outlet_Establishment_Year" to datetime
        data['Outlet_Establishment_Year'] = pd.to_datetime(data['Outlet_Establishment_Year'], format='%Y')

        # Extract the year from the datetime column
        current_year = datetime.now().year
        data['Outlet_Years'] = current_year - data['Outlet_Establishment_Year'].dt.year

        # Drop the original "Outlet_Establishment_Year" column if needed
        data.drop('Outlet_Establishment_Year', axis=1, inplace=True)

        return data

        
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        
        try:
            data = pd.read_csv(Path(self.ingestion_config.raw_data_path))
            logging.info(" i have read dataset as a df")
            
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(" i have saved the raw dataset in artifact folder")
            
            logging.info("here i have performed train test split")
            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion part completed")
            
            return (
                 
                
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
           logging.info("exception during occured at data ingestion stage")
           raise customexception(e,sys)
    