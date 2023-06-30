import os
import sys
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

## Function to Save Pickel File
def save_object(file_path,obj):
    '''
    This Function Will Save Pickel File

    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error Occured In Save Object Function")
        raise CustomException(e, sys)


def model_evaluation(X_train,y_train,X_test,y_test,models,param):
    '''
    This Method Will Train And Evaluate the model on hypermaters

    '''
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            rs= RandomizedSearchCV(model, para,cv=3)
            rs.fit(X_train,y_train)

            model.set_params(**rs.best_params_)
            ## Trani Model
            model.fit(X_train,y_train)

            ## Make Prediction
            y_test_pred = model.predict(X_test)

            ## Get The Accuracy using R2 Score
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
            logging.info("Error Occured In Model Traning")
            raise CustomException(e, sys)


def load_object(file_path):
    '''
    This Function Will Load Pickel File

    '''
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error Occured In Loding Object Function")
        raise CustomException(e, sys)

