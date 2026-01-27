import numpy as np 
import pandas as pd
import os 
import sys , dill
from src.exception import customException


def save_object(file_path , obj):
    try:
        # 1) Extract the directory from the full file path
        # 2) Create that directory if it doesn’t exist
        # 3)THEN save the file safely

        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj , file_obj)
    
    except  Exception as e:
        raise customException(e , sys)

