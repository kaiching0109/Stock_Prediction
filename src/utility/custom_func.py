"""
Here contains various custom functions to import.
"""
import os
import pandas as pd
import datetime

"""
@Param: String file_path
@Return Pandas.DataFrame
"""
def read_csv(file_path: str) -> pd.DataFrame:
     print("MESSAGE: Loaded data from " + file_path)
     try:
         df = pd.read_csv(file_path)
         print("SUCCESS: DataFrame Generated")
         return df
     except:
        print("FAIL: DataFrame CANNOT be Generated")
        return None

def output_csv(df: pd.DataFrame, filename: str, file_path: str = "../../../data/processed/")->pd.DataFrame:
    try:
        df.to_csv(file_path+filename)
        print("SUCCESS: CSV ", filename,  " Generated")
        return True
    except Exception as e:
        print("FAIL: CSV ", filename,  " CANNOT be Generated")
        print(e)
        return False


def to_date(date_int: int) -> datetime:
    return datetime.datetime.strptime(str(date_int),'%Y%m%d')

if __name__ == '__main__':
    #testing purpose
    df = pd.read_csv("../../data/raw/data.csv")
    output_csv(df, "processed_data.csv")
