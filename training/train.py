from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import argparse
import os

import numpy as np
import pandas as pd


import joblib

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):
    # Clean data, apply one hot encoding to categorical columns
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)

    return x_df, y_df

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(path)

x, y = clean_data(ds)

# TODO: Split data into train and test sets.

test_size = 0.25
random_state = 7

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    output_folder='./outputs'
    os.makedirs(output_folder, exist_ok=True)    
    joblib.dump(model, "./outputs/saved_model.joblib")

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

