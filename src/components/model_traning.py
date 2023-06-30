import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.utils import model_evaluation,save_object

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


@dataclass
class ModelTraningConfig:
    traning_model_file_obj = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_traner_config = ModelTraningConfig()

    def initated_model_traning(self,train_array,test_array):

        '''
        This Method Will Train Multipal Models Aand Evaluate

        '''
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Elastic":ElasticNet(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(n_estimators=400,max_depth=30,min_samples_split=5,min_samples_leaf=4),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(algorithm="ball_tree")
            }
            params = {
                "LinearRegression":{
                    
                },
                "Ridge":{
                    "alpha": [0.01, 0.1, 1, 10]
                    
                },
                "Lasso":{
                    "alpha": [0.01, 0.1, 1, 10]
                },
                "Elastic":{
                    "alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.2, 0.4, 0.6, 0.8]
                },
                "DecisionTreeRegressor":{
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter":['best','random'],
                    "max_depth": [8,15,20,25],
                    "min_samples_split": [8,10,5],
                    "min_samples_leaf": [5,8],
                    "max_features":["auto","sqrt","log2"]
                },
                "RandomForestRegressor":{

                },
                "AdaBoostRegressor":{
                    'n_estimators': [200,300],
                    "loss":["linear", "square", "exponential"],
                    "learning_rate":[1,0.1,0.01,0.001],
                },
                "GradientBoostingRegressor":{
                    'n_estimators': [400,500],
                    "learning_rate":[0.1],
                },
                "KNeighborsRegressor":{
                    "n_neighbors":[8,10,12]
                }

            }
            model_report:dict=model_evaluation(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)

                ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_object(file_path=self.model_traner_config.traning_model_file_obj,
                obj = best_model
                )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)


            









        except Exception as e:
            logging.info("Error Occured In Model Traning Stage")
            raise CustomException(e, sys)




