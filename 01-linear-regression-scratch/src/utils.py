from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path

def fetch_dataset():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    df.to_csv('data/california_housing.csv', index=False)

def split_dataset(df, train_pt=60, cv_pt=20):
    train_index = (len(df) * train_pt) / 100
    cv_index = (len(df) * cv_pt) / 100

    train_df = df[:train_index]
    cv_df = df[train_index:train_index+cv_index]
    test_df = df[train_index+cv_index:]

    return train_df, cv_df, test_df