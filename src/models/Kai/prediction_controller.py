"""
Here is to make predictions.
"""

"""
SET UP RELATIVE PATH
"""
import sys
import os
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(os.path.dirname(fileDir))
feature_path = os.path.join(rootDir, 'features')
vizualizer_path = os.path.join(rootDir, 'visualization')
# data_path = os.path.join(rootDir, 'data')
sys.path.extend([feature_path, vizualizer_path])

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from feature_builder import raw_data_processor, feature_scaling
from vizualizer import visualizeSimpleLinearRegreesionResult
from sklearn.metrics import mean_squared_error, r2_score
from netural_network import netural_network
import pandas as pd
from matplotlib import pyplot
# from simple_regression import simple_regression
from train_controller import train_controller

class prediction_controller:
    def __init__(self, data_set:tuple):
        self.X_train, self.X_test, self.Y_train, self.Y_test = data_set
        self.result = {}

    def set_data_set(self, data_set:tuple):
        self.X_train, self.X_test, self.Y_train, self.Y_test = data_set

    def predict_with_simple_regression(self):
        print("MESSAGE: Predicting with simple regression")
        try:
            X_train = self.X_train[:,1].reshape(-1, 1) #Get training Price only
            X_test = self.X_test[:, 1].reshape(-1, 1) #Get testing Price only
            print(X_train)
            print()
            print(X_test)
            regr = linear_model.LinearRegression() # Create linear regression object
            regr.fit(X_train, self.Y_train) # Train the model using the training sets
            Y_pred_train = regr.predict(X_train) # Make predictions using the training set
            Y_pred_test = regr.predict(X_test) # Make predictions using the testing set
            self.result["simple_regression"] = {
                "coefficients": regr.coef_,
                "mean_squared_error": mean_squared_error(self.Y_test, Y_pred_test),
                "r-squared": r2_score(self.Y_test, Y_pred_test),
                "x_train": X_train,
                "y_train": self.Y_train,
                "x_test": X_test,
                "y_test": self.Y_test,
                "y_pred_train": Y_pred_train,
                "y_pred_test": Y_pred_test
            }
            print("SUCCESS: Result predcited with regression is generated.")
            print("coefficients: ", self.result["simple_regression"]["coefficients"])
            print("mean squared error: ", self.result["simple_regression"]["mean_squared_error"])
            print("r-squared: ", self.result["simple_regression"]["r-squared"]) # good fit (high  𝑅2 )
            self.calcResiduals()
        except Exception as e:
            self.result["simple_regression"] = None
            print("FAIL: CANNOT predict with regression.")
            print(e)

    def calcResiduals(self):
        residuals = [self.result["simple_regression"]["y_test"][i]-self.result["simple_regression"]["y_pred_test"][i] for i in range(len(self.result["simple_regression"]["y_pred_test"]))]
        residuals = pd.DataFrame(residuals)
        residuals.plot()
        pyplot.show()
        print(residuals.describe())

    def predict_with_netural_network(self):
        try:
            print("MESSAGE: Predicting with RNN")
            print(self.X_train)
            # print(self.X_test)
            netural_network_controller = netural_network(self.X_train, self.Y_train, self.X_test, self.Y_test)
            netural_network_controller.compile()
            Y_pred_train, Y_pred_test = netural_network_controller.predict()
            print(Y_pred_train, Y_pred_test)
            self.result["rnn"] = {
                "y_pred_train": Y_pred_train,
                "y_pred_test": Y_pred_test
            }
            print("SUCCESS: Result predcited with RNN is generated.")
        except Exception as e:
            self.result["rnn"] = None
            print("FAIL: CANNOT predict with RNN.")
            print(e)

    def get_result(self):
        return self.result

if __name__ == '__main__':
    df = raw_data_processor()
    trainController = train_controller(df)
    data_set = trainController.get_processed_data_set()
    predictionController = prediction_controller(data_set)

    """
    predict_with_simple_regression
    """
    predictionController.predict_with_simple_regression()
    result = predictionController.get_result()
    X_train = result["simple_regression"]["x_train"]
    X_test = result["simple_regression"]["x_test"]
    Y_train = result["simple_regression"]["y_train"]
    Y_test = result["simple_regression"]["y_test"]
    Y_pred_train = result["simple_regression"]["y_pred_train"]
    Y_pred_test = result["simple_regression"]["y_pred_test"]
    visualizeSimpleLinearRegreesionResult(X_train, Y_train, Y_pred_train,
        "Signal", "Price", "Price vs Signal (Training)") #visualizing training result
    visualizeSimpleLinearRegreesionResult(X_test, Y_test, Y_pred_test,
        "Signal", "Price", "Price vs Signal (Testing)") #visualizing test result

    """
    predict_with_netural_network
    """
    # predictionController.predict_with_netural_network()
    # result = predictionController.get_result()
    # X_train, X_test, Y_train, Y_test = data_set
    # Y_pred_train = result["rnn"]["y_pred_train"]
    # Y_pred_test = result["rnn"]["y_pred_test"]
    # print(Y_pred_train)
    # print(Y_train)
    # X_train = X_train[:,1].reshape(-1, 1) #Get training Price only
    # X_test = X_test[:, 1].reshape(-1, 1) #Get testing Price only
    # visualizeSimpleLinearRegreesionResult(X_train, Y_train, Y_pred_train,
    #     "Price", "Signal", "Price vs Signal (Training)") #visualizing training result
    # visualizeSimpleLinearRegreesionResult(X_test, Y_test, Y_pred_test,
    #     "Price", "Signal", "Price vs Signal (Testing)") #visualizing test result
