"""
@Kai
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
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot


"""
@ Kai
Shows processed data through pandas
"""
def visualizeProcessedDataFrame() -> pd.DataFrame:
    PROCESSED_DATA_FILENAME = "data.csv"
    df = read_csv(data_path+"/processed/data.csv")
    return df

"""
@ Kai
"""
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

"""
@ Kai
For visualizing Simple Linear Regreesion Result (Graph)
"""
def visualizeSimpleLinearRegreesionResult(x: np.ndarray,
                            y: np.ndarray,
                            y_pred: np.ndarray,
                            x_label: str=None,
                            y_label: str=None,
                            title: str=None,
                            filename: str=None,
                            path: str=None):
    plt.scatter(x, y,  color='red')
    plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    if(title):
        plt.title(title)
    if(x_label and y_label):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    if(filename):
        try:
            plt.savefig(figure_path + "/" + filename)
            print("SUCCESS: ", filename, " is Generated")
        except:
            print("FAIL: ", filename, " CANNOT be Generated")
    else:
        plt.show()

def visualizeResiduals(result: dict):
    print("residuals: ")
    # residuals.plot()
    # plt.show()
    print(result["residuals"].describe())
    # Instantiate the linear model and visualizer
    ridge = Ridge()
    visualizer = ResidualsPlot(ridge)

    visualizer.fit(result["x_train"], result["y_train"])  # Fit the training data to the model
    visualizer.score(result["x_test"], result["y_test"])  # Evaluate the model on the test data
    visualizer.poof()                 # Draw/show/poof the dat

"""
@ Kai
For getting regression report (text), please call this function
"""
def displayRegressionReport(regression_report: dict):
    print("coefficients: ", regression_report["coefficients"])
    print("intercept: ", regression_report["intercept"])
    print("mean squared error: ", regression_report["mean_squared_error"])
    print("r-squared: ", regression_report["r-squared"]) # good fit (high  𝑅2 )
    visualizeResiduals(regression_report)

if __name__ == '__main__':
    df = visualizeProcessedDataFrame()
    #visualizeProcessedData(df, "date", "spy_close_price", "Date", "Close Price", "price_date.png")
    #visualizeProcessedData(df, "date", "signal", "Date", "Signal", "signal_date.png")
    #visualizeProcessedData(df, "spy_close_price", "signal", "Price", "Signal", "signal_price.png")
    # visualizePredictResult()
