"""
Here is to transform raw data into features for modeling.
"""

"""
SET UP RELATIVE PATH
"""
import sys
import os
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(os.path.dirname(fileDir))
utility_path = os.path.join(rootDir, 'src', 'utility')   # Get the directory for StringFunctions
data_path = os.path.join(rootDir, 'data')
sys.path.extend([utility_path,data_path])

"""
IMPORTS
"""
from custom_func import read_csv,to_date,output_csv
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

"""
Read raw data -> Restructure data -> Clean data -> Sort data by date ->
Generate processed data -> Return Processed DataFrame

@return: pandas.core.frame.DataFrame
"""
def raw_data_processor():
    DATE_COLUMN_NAME = "date"
    PROCESSED_DATA_FILENAME = "data.csv"
    df = read_csv(data_path+"/raw/data.csv")
    try:
        df = date_to_numeric(df)
        print("MESSAGE: DATA CLEANING started")
        print("MESSAGE: data row before cleaning: " + str((df.shape)[0]))
        df = clean_data(df)
        # df = df.sort_values(DATE_COLUMN_NAME)
        print("MESSAGE: data row before cleaning: " + str((df.shape)[0]))
        print("SUCCESS: DATA is CLEAN")
    except Exception as e:
        print("FAIL: Error occured during the DATA CLEANING process")

    if(output_csv(df, PROCESSED_DATA_FILENAME, data_path + "/processed")):
        print("SUCCESS: processed data file Generated")
    else:
        print("FAIL: processed data file fail to Generated")
    return df

"""
@return: pandas.core.frame.DataFrame
"""
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    df.fillna(df.mean())
    df = clean_outliner(df)
    return df

def date_to_numeric(df: pd.DataFrame):
    DATE_COLUMN_NAME = "date"
    # date = df[DATE_COLUMN_NAME].map(lambda x: datetime.datetime.strptime(str(x), '%H:%M:%S'))
    date = df[DATE_COLUMN_NAME] = df.date.map(lambda x: to_date(x))
    df[DATE_COLUMN_NAME] = [i.timestamp() for i in date]
    return df

"""
Detect and remove outliers
@return: pandas.core.frame.DataFrame
"""
def clean_outliner(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

"""
@return: pandas.core.frame.DataFrame
"""
def feature_scaling(train_set: pd.DataFrame) -> pd.DataFrame:
    # sc = StandardScaler()
    # train_set_scaled = sc.fit_transform(train_set)
    # test_set_scaled = sc.transform(test_set)
    # return train_set_scaled, test_set_scaled
    sc = MinMaxScaler(feature_range = (0, 1))
    train_set_scaled = sc.fit_transform(train_set)
    return train_set_scaled

"""
@param: String start, String end
Both in foramt of YYYYMMDD
@return: pandas.core.frame.DataFrame
"""
def get_historical_data(df, start: int, end:int=None):
    DATE_COLUMN_NAME = "date"
    if not(end):
        end = datetime.datetime.now()
    else:
        end = to_date(end)
    start = to_date(start)
    date_series = df[DATE_COLUMN_NAME]
    date_mask = (date_series > start) & (date_series <= end)
    return df.loc[date_mask]

if __name__ == '__main__':
    #testing purpose
    df = raw_data_processor()
