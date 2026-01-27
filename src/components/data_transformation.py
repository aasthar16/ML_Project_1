import os
import sys
import pickle
from src.exception import customException
from src.logger import logging
import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object


@dataclass
class DataTransfromationConfig:
    preproccesor_obj_file_path=os.path.join('artifact', 'preprocessor.pkl')
    # this the path to store our model in pickle file 

class Data_Transformation:
    def __init__(self):
        self.data_transfromation_config=DataTransfromationConfig()


# this fn is to transform data
    def get_data_transformer_object(self):
        try:
            numerical_columns=["writing_score" , "reading_score"]
            categorical_columns=["gender",
                                 "race_ethnicity",
                                 "parental_level_of_education",
                                 "lunch",
                                 "test_preparation_course"
                                 ]
            # imputer is used to handle missing values
            num_pipeline=Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy='median')),
                    ("scaler" , StandardScaler())
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder" , OneHotEncoder())
                ]
            )

            logging.info("Categorical Columns Encoding started")
            logging.info("Munerical Columns Standard Scaling started")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline" ,num_pipeline, numerical_columns),
                    ("cat_pipeline" , categorical_pipeline , categorical_columns)
                ]
            )   

            logging.info("Categorical Columns Encoding completed")
            logging.info("Munerical Columns Standard Scaling completed")

            return preprocessor

        except Exception as e:
            raise customException(e, sys)
        


    def initiate_data_tranformation(self ,train_path , test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            prep=self.get_data_transformer_object()

            target_col_name="math_score"
            numerical_columns=["writing_score" , "reading_score"]
            categorical_columns=["gender",
                                 "race_ethnicity",
                                 "parental_level_of_education",
                                 "lunch",
                                 "test_preparation_course"
                                 ]
            
            input_feature_train_df=train_df.drop(columns=[target_col_name] , axis=1)
            target_feature_train_df=train_df[target_col_name]

            input_feature_test_df=test_df.drop(columns=[target_col_name] , axis=1)
            target_feature_test_df=test_df[target_col_name]

            logging.info("Applying preprocessing object on traininf and testing df")

            input_feature_train_arr=prep.fit_transform(input_feature_train_df)
            input_feature_test_arr=prep.transform(input_feature_test_df)
           

        #    c - combining  (I.Imp)
            train_arr = np.c_[
                input_feature_train_arr , np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]

            # saving the pkl file in hard disk

            save_object(
                file_path=self.data_transfromation_config.preproccesor_obj_file_path,
                obj=prep
            )
            return (
                train_arr,
                test_arr,
                self.data_transfromation_config.preproccesor_obj_file_path  #pkl file path 

            )
        except Exception as e:
            raise customException(e , sys)
