import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    This Class Will Configure THe Preprocessor Object

    '''
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This Method Will Give Data Transformation Object

        '''
        try:
            logging.info("Data Transformation Initated")

            numerical_features = ['CabinType', 'EntertainmentRating', 'FoodRating', 'GroundServiceRating',
            'Recommended', 'SeatComfortRating', 'ServiceRating', 'TravelType',
            'ValueRating', 'WifiRating']

            # Create Numerical Pipline
            # numeric Pipline
            num_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler()),
            ]
        )

            # Create preprocessor Object
            preprocessor = ColumnTransformer([
                ("num_pipline",num_pipline,numerical_features)
            ])

            return preprocessor
            logging.info("Pipline Complited")


        except Exception as e:
            logging.info("Error Occured In Data Transformation Stage")
            raise CustomException(e, sys)


    def initited_data_transformation(self,train_path,test_path):
        '''
        This Method Will Take Preprocessor Object and Transform Data

        '''
        try:
            ## Read Data Frame Train And Test
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning And Testing Data Complited")
            logging.info(f"Traning Data head: \n{train_data.head().to_string()}")
            logging.info(f"Testing Data head: \n{test_data.head().to_string()}")

            logging.info("Obtaning Preprocessor Object")

            preprocessor_obj = self.get_data_transformation_object()

            target_columns = "OverallRatings"
            drop_columns = [target_columns]

            ## Split dependent and Independent Features
            input_features_train_data = train_data.drop(drop_columns,axis=1)
            target_feature_train_data = train_data[target_columns]

            ## Split dependent and Independent Features
            input_features_test_data = test_data.drop(drop_columns,axis=1)
            target_feature_test_data = test_data[target_columns]

            ## Apply Preprocessor Object
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_data)

            logging.info("Apply Preprocessor Object on Train And Test Data")

            ## Convert in to Array To Become Fast
            train_array = np.c_[input_features_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_features_test_arr,np.array(target_feature_test_data)]


            ## Calling Save Object Function To Save Pickel File
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
             obj = preprocessor_obj)

            logging.info("preprocessor Object File Is Save")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Occured In Data Transformation Stage")
            raise CustomException(e, sys)



   


