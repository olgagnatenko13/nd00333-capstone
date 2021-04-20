import joblib
import json 
import os 
from azureml.core.model import Model
import pandas as pd 

def init():
    global model
    model_path = Model.get_model_path("attrition-model")
    print("Model Path is  ", model_path)
    model = joblib.load(model_path)

def run(data):
    try:
        raw_data = json.loads(data)
        df = pd.DataFrame(raw_data["data"])
        result = model.predict(df)
        return {'data' : result.tolist() , 'message' : "Successfully classified attrition"}
    except Exception as e:
        error = str(e)
        return {'data' : error , 'message' : 'Failed to classify attrition'}