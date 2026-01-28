import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import customException
from src.logger import logging
from src.utils import evaluate_model , save_object

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def inititate_model_trainer(self, training_array , test_array ):
        try:
            logging.info("Splitting training and test input data. ")
            X_train , y_train , X_test , y_test=(
                training_array[:, :-1],
                training_array[: , -1],
                test_array[: , :-1],
                test_array[:, -1]
            )
            models ={
                "Linear Regression" : LinearRegression(),
                "CatBoost Regressor"   : CatBoostRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor() ,
                "Random Forest Regressor":RandomForestRegressor(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "XGBRegressor":XGBRegressor()

            }

            model_report:dict=evaluate_model(X_train=X_train , y_train=y_train , X_test=X_test, y_test=y_test, models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_models=models[best_model_name]

            if best_model_score<0.6:
                raise customException("No best model found")
            
            logging.info("Best model found.")

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_models

            )

            predicted=best_models.predict(X_test)

            score=r2_score(y_test ,predicted)

            return score
        except Exception as e:
            raise customException(e ,sys)


