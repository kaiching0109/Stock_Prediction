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
netural_network_path = os.path.join(fileDir, 'netural_network')
# data_path = os.path.join(rootDir, 'data')
sys.path.extend([feature_path, vizualizer_path, netural_network_path])

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from feature_builder import raw_data_processor, feature_scaling
from vizualizer import visualizeSimpleLinearRegreesionResult
from sklearn.metrics import mean_squared_error, r2_score

import netural_network
import simple_regression
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
            regr = linear_model.LinearRegression() # Create linear regression object
            regr.fit(self.X_train, self.Y_train) # Train the model using the training sets
            Y_pred_train = regr.predict(self.X_train) # Make predictions using the training set
            Y_pred_test = regr.predict(self.X_test) # Make predictions using the testing set
            self.result["simple_regression"] = {
                "coefficients": regr.coef_,
                "mean_squared_error": mean_squared_error(self.Y_test, Y_pred_test),
                "r-squared": r2_score(self.Y_test, Y_pred_test),
                "y_pred_train": Y_pred_train,
                "y_pred_test": Y_pred_test
            }
            print("SUCCESS: Result predcited with regression is generated.")
        except Exception as e:
            self.result["simple_regression"] = None
            print("FAIL: CANNOT predict with regression.")
            print(e)

    def predict_with_multi_regression(self):
        print("MESSAGE: Predicting with multi regression")


    def get_result(self):
        return self.result

if __name__ == '__main__':
    df = raw_data_processor()
    trainController = train_controller(df)
    data_set = trainController.get_processed_data_set()
    predictionController = prediction_controller(data_set)
    predictionController.predict_with_simple_regression()
    result = predictionController.get_result()
    X_train, X_test, Y_train, Y_test = data_set
    Y_pred_train = result["simple_regression"]["y_pred_train"]
    Y_pred_test = result["simple_regression"]["y_pred_test"]
    visualizeSimpleLinearRegreesionResult(X_train, Y_train, Y_pred_train,
        "X_train", "Y_train", "X vs Y (Training)") #visualizing training result
    visualizeSimpleLinearRegreesionResult(X_test, Y_test, Y_pred_test,
        "X_test", "Y_test", "X vs Y (Testing)") #visualizing test result
