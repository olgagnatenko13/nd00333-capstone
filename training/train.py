from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import argparse
import os

import numpy as np
import pandas as pd

import joblib
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):

    def encode_categorical_column(df, column_name):
        encoded_columns = pd.get_dummies(df[column_name], prefix=column_name)
        encoded_columns = encoded_columns.drop(encoded_columns.columns[[0]], axis=1)
        df.drop(column_name, inplace=True, axis=1)
        df = df.join(encoded_columns)  
        return df 

    # Clean data, apply one hot encoding to categorical columns
    x_df = data.to_pandas_dataframe().dropna()
    x_df.drop(["EmployeeNumber", "EmployeeCount"], inplace=True, axis=1)

    x_df = encode_categorical_column(x_df, "BusinessTravel")
    x_df = encode_categorical_column(x_df, "Department")
    x_df = encode_categorical_column(x_df, "EducationField")

    x_df["Gender"] = x_df.Gender.apply(lambda s: 1 if s == "Male" else 0)
    x_df["OverTime"] = x_df.OverTime.apply(lambda s: 1 if s == "Yes" else 0)

    x_df = encode_categorical_column(x_df, "JobRole")
    x_df = encode_categorical_column(x_df, "MaritalStatus")

    y_df = x_df.pop("Attrition").apply(lambda s: 1 if s == "Yes" else 0)

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_df)
    x_df = pd.DataFrame(x_scaled, columns=x_df.columns)

    return x_df, y_df

path = "https://raw.githubusercontent.com/olgagnatenko13/nd00333-capstone/master/dataset/Dataset_for_Classification.csv"
ds = TabularDatasetFactory.from_delimited_files(path)

x, y = clean_data(ds)

# Split data into train and test sets.

test_size = 0.25
random_state = 7

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Regularization strength: greater values cause stronger regularization")
    parser.add_argument('--gamma', type=float, default=0.1, help="How far the influence of a single training example reaches; greater values cause greater reach")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Gamma:", np.float(args.gamma))

    kernel = "rbf" # RBD kernel used for the model 
    max_iter = 100

    model = SVC(kernel = kernel, C = args.C, gamma = args.gamma, max_iter = max_iter).fit(x_train, y_train)

    output_folder='./outputs'
    os.makedirs(output_folder, exist_ok=True)    
    joblib.dump(model, "./outputs/SVC_model_with_hyperparameters.joblib")

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_pred, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

