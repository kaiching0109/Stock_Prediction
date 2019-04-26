"""
Here is to train models.
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
# data_path = os.path.join(rootDir, 'data')
sys.path.extend([feature_path])

import pandas as pd
import numpy as np
from feature_builder import raw_data_processor, feature_scaling
from sklearn.model_selection import train_test_split

class train_controller:
    def __init__(self, df: pd.DataFrame):
        self.processed_data = df.values
        self.initialize_data()

    def initialize_data(self):
        print("MESSAGE: Initializing training and test data...")
        SPY_CLOSE_COLUMN_NAME = "spy_close_price"
        SIGNAL_COLUMN_NAME = "signal"
        try:
            n_samples, n_features = self.processed_data.shape
            X = self.processed_data[:, np.newaxis, 1]
            Y = self.processed_data[:, np.newaxis, 2]
            # print(Y)
            # train_size = round(size * 0.80)
            # self.train_data = self.processed_data[:train_size]
            # self.test_data = self.processed_data[train_size:]
            # train_set_scaled, test_set_scaled = feature_scaling(self.train_data, self.test_data)

            # X = self.processed_data[SIGNAL_COLUMN_NAME]
            # Y = self.processed_data[SPY_CLOSE_COLUMN_NAME]
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                                                X, Y, test_size=.2, random_state=0)
            print("SUCCESS: Initialized training and test data")
        except Exception as e:
            print("FAIL: Unable to initialize training and test data")
            print(e)



    def get_processed_data_set(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

if __name__ == '__main__':
    df = raw_data_processor()
    controller = train_controller(df)
