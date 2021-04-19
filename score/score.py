import joblib
import json 
from azureml.core.model import Model
from azureml.data.dataset_factory import TabularDatasetFactory
import os 

def clean_data(data):

    def encode_categorical_column(df, column_name):
        encoded_columns = pd.get_dummies(df[column_name], prefix=column_name)
        encoded_columns = encoded_columns.drop(encoded_columns.columns[[0]], axis=1)
        df.drop(column_name, inplace=True, axis=1)
        df = df.join(encoded_columns)  
        return df 

    # Clean data, apply one hot encoding to categorical columns
    x_df = pd.DataFrame(data)
    print("df", x_df)
    x_ddf = x_df.dropna()
    x_df = encode_categorical_column(x_df, "BusinessTravel")
    x_df = encode_categorical_column(x_df, "Department")
    x_df = encode_categorical_column(x_df, "EducationField")
    x_df["Gender"] = x_df.Gender.apply(lambda s: 1 if s == "Male" else 0)
    x_df = encode_categorical_column(x_df, "JobRole")
    x_df = encode_categorical_column(x_df, "MaritalStatus")
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_df)
    x_df = pd.DataFrame(x_scaled, columns=x_df.columns)

    return x_df

def init():
    global model
    model_path = Model.get_model_path("attrition-model")
    print("Model Path is  ", model_path)
    model = joblib.load(model_path)

def run(data):
    try:
        raw_data = json.loads(data)
        print("raw", raw_data.data)
        correct_data = clean_data(raw_data.data)
        result = model.predict(correct_data)
        return {'data' : result.tolist() , 'message' : "Successfully classified attrition"}
    except Exception as e:
        error = str(e)
        return {'data' : error , 'message' : 'Failed to classify attrition'}