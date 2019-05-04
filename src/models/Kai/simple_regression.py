"""
@ Kai
Here does all simple regression work
"""

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
import pandas as pd

class simple_regression:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test
        # dataset = pd.DataFrame({"X_train": X_train[:,0], "Y_train": Y_train[:, 0]}, index=list(range(len(X_train[:] - 1))), columns=['X_train', 'Y_train'])
        # print(dataset)
        # self.X_train = self.X_train[:, 0]
        # self.X_test = self.X_test[:, 0]
        # self.Y_train = self.X_test[:, 1]
        # self.Y_test = self.X_test[:, 1]
        self.regr = linear_model.LinearRegression() # Create linear regression object

    def compile(self):
        self.regr.fit(self.X_train, self.Y_train) # Train the model using the training sets

    def predict(self):
        Y_pred_train = self.regr.predict(self.X_train) # Make predictions using the training set
        Y_pred_test = self.regr.predict(self.X_test) # Make predictions using the testing set
        return {
            "coefficients": self.regr.coef_,
            "intercept": self.regr.intercept_,
            "mean_squared_error": mean_squared_error(self.Y_test, Y_pred_test),
            "r-squared": r2_score(self.Y_test, Y_pred_test),
            "x_train": self.X_train,
            "y_train": self.Y_train,
            "x_test": self.X_test,
            "y_test": self.Y_test,
            "y_pred_train": Y_pred_train,
            "y_pred_test": Y_pred_test
        }

    def getResiduals(self, result):
        residuals = [result["y_test"][i]-result["y_pred_test"][i] for i in range(len(result["y_pred_test"]))]
        return pd.DataFrame(residuals)
