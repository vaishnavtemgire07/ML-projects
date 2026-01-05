import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from source.exception import CustomException
from source.logger import logging
import os
from source.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['id','carat','depth','table','x','y','z']
            categorical_columns = ['cut','color','clarity']
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                
            ])
            # Instantiate OneHotEncoder in a way that's compatible with different sklearn versions
            try:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:
                # older sklearn versions use 'sparse' argument
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', ohe),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)       
            pass
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Try with default delimiter (comma); if 'price' missing, retry with tab separator
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if 'price' not in train_df.columns:
                logging.warning("Target column 'price' not found with default parser; retrying with tab separator for train file")
                train_df = pd.read_csv(train_path, sep='\t')

            if 'price' not in test_df.columns:
                logging.warning("Target column 'price' not found with default parser; retrying with tab separator for test file")
                test_df = pd.read_csv(test_path, sep='\t')

            logging.info(f"Train columns: {train_df.columns.tolist()}")
            logging.info(f"Test columns: {test_df.columns.tolist()}")
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")

            # Final check
            if 'price' not in train_df.columns or 'price' not in test_df.columns:
                raise CustomException(Exception(f"Target column 'price' not found. Train cols: {train_df.columns.tolist()}, Test cols: {test_df.columns.tolist()}"), sys)
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name = 'price'
            numerical_columns = ['id','carat','depth','table','x','y','z']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessor object on training and testing data")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessor object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
            pass
        