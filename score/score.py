import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("best_automl_model")
    print("Model Path is  ", model_path)
    model = joblib.load(model_path)

def run(data):
    try:
        print(data)
        result = model.predict(data['data'])
        return {'data' : result.tolist() , 'message' : "Successfully classified attrition"}
    except Exception as e:
        error = str(e)
        return {'data' : error , 'message' : 'Failed to classify attrition'}