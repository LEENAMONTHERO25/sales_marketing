import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime 

from dataclasses import dataclass
from src.Sales_Marketing.exception import customexception
from src.Sales_Marketing.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.Sales_Marketing.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

        
    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')
            categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                               'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
            numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years'
                             ]
            
            Item_Fat_Content_catagories=['Low Fat','low fat','LF', 'Regular','reg']

            Item_Type_catagories=['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
                                  'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
                                  'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
                                  'Breads', 'Starchy Foods', 'Others', 'Seafood']
            
            Outlet_Identifier_catagories=['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027',
                                          'OUT035', 'OUT045', 'OUT046', 'OUT049']
            
            Outlet_Size_catagories =['Small','Medium','High']

            Outlet_Location_Type_categories=['Tier 1', 'Tier 2', 'Tier 3']

            Outlet_Type_categories=[ 'Grocery Store','Supermarket Type1', 'Supermarket Type2', 
                                     'Supermarket Type3']
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Item_Fat_Content_catagories,
                                                             Item_Type_catagories,
                                                             Outlet_Identifier_catagories,
                                                             Outlet_Size_catagories,
                                                             Outlet_Location_Type_categories,
                                                             Outlet_Type_categories])),
                ('scaler',StandardScaler())
                ]

            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
            

            
            
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
            
    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
              # Convert 'Outlet_Establishment_Year' to datetime
            train_df['Outlet_Establishment_Year'] = pd.to_datetime(train_df['Outlet_Establishment_Year'], format='%Y')
            test_df['Outlet_Establishment_Year'] = pd.to_datetime(test_df['Outlet_Establishment_Year'], format='%Y')


            # Assuming "Outlet_Years" is already created during data ingestion
            current_year = datetime.now().year
            logging.info("Adding Outlet_Years feature to the dataset.")
            train_df['Outlet_Years'] = current_year - train_df['Outlet_Establishment_Year'].dt.year
            test_df['Outlet_Years'] = current_year - test_df['Outlet_Establishment_Year'].dt.year


            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name = "Item_Outlet_Sales"
            drop_columns = [target_column_name,'Item_Identifier','Outlet_Establishment_Year']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
            