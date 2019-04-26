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

class train_controller:
    def __init__(self, df: pd.DataFrame):
        print("train_controller")
        self.processed_data = df

    def initialize_data(self):
        size = (self.processed_data.shape)[0]
        train_size = round(size * 0.80)
        self.train_data = self.processed_data[:train_size]
        self.test_data = self.processed_data[train_size:]



if __name__ == '__main__':
    controller = train_controller()
