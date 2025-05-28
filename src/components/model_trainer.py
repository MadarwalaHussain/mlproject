# Model Training step

import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import setup_logger, get_logger
from src.utils import save_object, evaluate_model

setup_logger()
logger = get_logger(__name__)

@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor()
                # "XGBRegressor": XGBRegressor(),
                # "CatBoostRegressor": CatBoostRegressor(verbose=False),
                # "AdaBoostRegressor": AdaBoostRegressor(),
                # "Gradient Boosting": GradientBoostingRegressor()
            }
            # Performing Hyper parameter tuning
            params = {
                "Decision Tree Regressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    # "max_depth": [None, 5, 10, 20],
                    # "max_features": [5,10, 15, 20, "auto", "sqrt", "log2"],
                    "splitter": ["best", "random"]
                },
                "Random Forest Regressor": {
                    "n_estimators": [10, 20, 50, 75, 100,200, 300],
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    # "max_depth": [None, 5, 10, 20],
                    # "max_features": ["auto", "sqrt", "log2"],
                },
                # "Gradient Boosting":{
                #     "n_estimators": [10, 20, 50, 75, 100,200, 300],
                #     "learning_rate": [0.01, 0.1, 0.2, 0.3],
                #     # "max_depth": [3, 5, 7, 9],
                #     "loss": ["ls", "lad", "huber", "quantile"],
                #     "criterion": ["friedman_mse", "squared_error"]
                # },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                # "XGBRegressor": {
                #     "n_estimators": [10, 20, 50, 75, 100,200, 300],
                #     "learning_rate": [0.01, 0.1, 0.2, 0.3],
                #     "max_depth": [3, 5, 7, 9],
                #     "booster": ["gbtree", "gblinear", "dart"], 
                # },
                # "CatBoostRegressor": {
                #     "iterations": [100, 200, 300],
                #     "learning_rate": [0.01, 0.1, 0.2, 0.3],
                #     "depth": [3, 5, 7, 9],
                #     "loss_function": ["RMSE", "MAE", "Quantile"]
                # },
                # "AdaBoostRegressor": {
                #     "n_estimators": [50, 100, 200],
                #     "learning_rate": [0.01, 0.1, 0.2, 0.3],
                #     "loss": ["linear", "square", "exponential"]
                # }

            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # To get the best score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logger.info(f"best model found is {best_model_name}")

            if best_model_score<0.6:
                raise CustomException("No Best Model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)