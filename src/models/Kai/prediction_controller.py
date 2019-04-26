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

from train_controller import train_controller
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from feature_builder import raw_data_processor, feature_scaling
from vizualizer import visualizePredictResult
from sklearn.metrics import mean_squared_error, r2_score

class prediction_controller:
    def __init__(self, data_set:tuple):
        self.X_train, self.X_test, self.Y_train, self.Y_test = data_set
        self.result = {}

    def predict_with_regression(self):
        print("MESSAGE: Predicting with regression")
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        print(self.X_train)
        print(self.Y_train)
        regr.fit(self.X_train, self.Y_train)
        # Make predictions using the testing set
        Y_pred = regr.predict(self.X_test)
        self.result["regression"] = {
            "coefficients": regr.coef_,
            "mean_squared_error": mean_squared_error(self.Y_test, Y_pred),
            "r-squared": r2_score(self.Y_test, Y_pred),
            "y_pred": Y_pred
        }

    def get_result(self):
        return self.result

if __name__ == '__main__':
    df = raw_data_processor()
    trainController = train_controller(df)
    data_set = trainController.get_processed_data_set()
    predictionController = prediction_controller(data_set)
    predictionController.predict_with_regression()
    result = predictionController.get_result()
    print(result)
    X_train, X_test, Y_train, Y_test = data_set
    Y_pred = result["regression"]["y_pred"]
    visualizePredictResult(X_test, Y_test, Y_pred)
