import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# inputs required by data ingestion , these inputs will be extracted using this class
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts' , 'train.csv')
    # this is the path in which we will store train_data

    test_data_path: str=os.path.join('artifacts' , 'test.csv')

    raw_data_path : str=os.path.join('artifacts' , 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('../../notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe.")
             
            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path) , exist_ok=True)

            df.to_csv(self.ingestionconfig.raw_data_path, index=False, header = True)
            logging.info("Train Test split initiated.")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestionconfig.train_data_path , index=False, header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path , index=False, header=True)

            logging.info('Ingestion of the data is completed.')

            return(
                self.ingestionconfig.train_data_path ,
                self.ingestionconfig.test_data_path 
            )
        except Exception as e:
            raise customException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()
        
    

