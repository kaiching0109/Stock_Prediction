from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model

class simple_regression():
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.regr = linear_model.LinearRegression() # Create linear regression object

    def complie(self):
        self.regr.fit(X_train, self.Y_train) # Train the model using the training sets

    def predict(self):
        Y_pred_train = self.regr.predict(self.X_train) # Make predictions using the training set
        Y_pred_test = self.regr.predict(self.X_test) # Make predictions using the testing set
        return {
            "coefficients": self.regr.coef_,
            "mean_squared_error": mean_squared_error(self.Y_test, Y_pred_test),
            "r-squared": r2_score(self.Y_test, Y_pred_test),
            "x_train": self.X_train,
            "y_train": self.Y_train,
            "x_test": self.X_test,
            "y_test": self.Y_test,
            "y_pred_train": Y_pred_train,
            "y_pred_test": Y_pred_test
        }

    def getResiduals(self):
        residuals = [self.result["simple_regression"]["y_test"][i]-self.result["simple_regression"]["y_pred_test"][i] for i in range(len(self.result["simple_regression"]["y_pred_test"]))]
        return pd.DataFrame(residuals)
