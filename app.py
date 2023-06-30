import pandas as pd
import numpy as np
import os
import sys

from flask import Flask,request,render_template,jsonify
from src.pipline.prediction_pipline import PredictPipline,CustomData

application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    '''
    This Function Wil Take Input From
    The Form and Predict the Output

    '''
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            CabinType = int(request.form.get("CabinType")),
            EntertainmentRating = int(request.form.get("EntertainmentRating")),
            FoodRating = int(request.form.get("FoodRating")),
            GroundServiceRating = int(request.form.get("GroundServiceRating")),
            Recommended = int(request.form.get("Recommended")),
            SeatComfortRating = int(request.form.get("SeatComfortRating")),
            ServiceRating = int(request.form.get("ServiceRating")),
            TravelType = int(request.form.get("TravelType")),
            ValueRating = int(request.form.get("ValueRating")),
            WifiRating = int(request.form.get("WifiRating")),
            )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.prediction(final_data)

        result = round(pred[0],1)

        return render_template("form.html",final_result = "Overall Arline Rating Is:{}".format(result))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
    


