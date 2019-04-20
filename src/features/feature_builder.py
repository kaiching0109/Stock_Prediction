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
from custom_func import read_csv
import pandas as pd

"""
@return: pandas.core.frame.DataFrame
"""
def raw_data_processor():
    print("raw_data_processor")

"""
@return: pandas.core.frame.DataFrame
"""
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("clean_data")

"""
@param: String start, String end
Both in foramt of YYYYMMDD
@return: pandas.core.frame.DataFrame
"""
def get_historical_data(start: str, end: str):
    print("get_historical_data")
    if not(end):
        end = datetime.now()

if __name__ == '__main__':
    #testing purpose
    # df = pd.read_csv("../../data/raw/data.csv")
    df = read_csv(data_path + "/raw/data.csv")
    clean_data(df)
