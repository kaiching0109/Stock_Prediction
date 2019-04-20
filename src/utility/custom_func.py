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
    print("read_csv")
    return pd.read_csv(file_path)

def output_csv(df: pd.DataFrame, filename: str, file_path: str = "../../data/processed/")->pd.DataFrame:
    print("read_csv")
    return df.to_csv(file_path+filename)

def to_date(date_string: str) -> datetime:
    return datetime.strptime(date_string,'%Y%m%d')

if __name__ == '__main__':
    #testing purpose
    df = pd.read_csv("../../data/raw/data.csv")
    output_csv(df, "processed_data.csv")
