"""
Here is used to create anything for visualizing.
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
figure_path = os.path.join(rootDir,'reports', 'figures')
sys.path.extend([utility_path,data_path])

from custom_func import read_csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def visualizeProcessedDataFrame() -> pd.DataFrame:
    PROCESSED_DATA_FILENAME = "data.csv"
    df = read_csv(data_path+"/processed/data.csv")
    return df

def visualizeProcessedData(df: pd.DataFrame, col_name_x: str,
                            col_name_y: str, x_label: str, y_label: str, filename: str=None):
    plt.figure(figsize = (18,9))
    plt.plot(df[col_name_x],df[col_name_y])
    plt.xticks(range(0,df.shape[0], 20),df[col_name_x].loc[::20],rotation=45)
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel(y_label,fontsize=18)
    if(filename):
        try:
            plt.savefig(figure_path + "/" + filename)
            print("SUCCESS: ", filename, " is Generated")
        except:
            print("FAIL: ", filename, " CANNOT be Generated")
    else:
        plt.show()

if __name__ == '__main__':
    df = visualizeProcessedDataFrame()
    visualizeProcessedData(df, "date", "spy_close_price", "Date", "Close Price", "price_date.png")
    visualizeProcessedData(df, "date", "signal", "Date", "Signal", "signal_date.png")
