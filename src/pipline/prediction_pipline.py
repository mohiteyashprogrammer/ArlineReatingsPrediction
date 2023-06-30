import os 
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object

class PredictPipline:
    def __init__(self):
        pass

    def prediction(self,features):
        '''
        This Method Will Be Predict The Output

        '''
        try:
            ## This line Of code Work in Any System
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            ## Lode Pickel File 
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            ## Apply Preprocessor Object And Scaled Data
            data_scaled = preprocessor.transform(features)

            ## Make prediction
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)

    
class CustomData:

    def __init__(self,
            CabinType:int,
            EntertainmentRating:int,
            FoodRating:int,
            GroundServiceRating:int,
            Recommended:int,
            SeatComfortRating:int,
            ServiceRating:int,
            TravelType:int,
            ValueRating:int,
            WifiRating:int
            ):

        self.CabinType = CabinType
        self.EntertainmentRating = EntertainmentRating
        self.FoodRating = FoodRating
        self.GroundServiceRating = GroundServiceRating
        self.Recommended = Recommended
        self.SeatComfortRating = SeatComfortRating
        self.ServiceRating = ServiceRating
        self.TravelType = TravelType
        self.ValueRating = ValueRating
        self.WifiRating = WifiRating

    def get_data_as_data_frame(self):
        '''
        This Function Will Create Custome Data Frame

        '''
        try:
            custome_data_input = {
                "CabinType":[self.CabinType],
                "EntertainmentRating":[self.EntertainmentRating],
                "FoodRating":[self.FoodRating],
                "GroundServiceRating":[self.GroundServiceRating],
                "Recommended":[self.Recommended],
                "SeatComfortRating":[self.SeatComfortRating],
                "ServiceRating":[self.ServiceRating],
                "TravelType":[self.TravelType],
                "ValueRating":[self.ValueRating],
                "WifiRating":[self.WifiRating]
            }

            data = pd.DataFrame(custome_data_input)
            logging.info("Data Frame Created")
            return data

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)
    
        