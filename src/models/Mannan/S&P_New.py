
# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

filepath = '/Users/manan/Desktop/S&P/data.csv'
stock = pd.read_csv(filepath)
print(stock.corr())
x = stock['signal'].iloc[:0].values.reshape(-1, 1)  
y = stock['spy_close_price'].iloc[:1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)  
x_train = stock['signal'][:-20]
x_test = stock['signal'][-20:]
y_train = stock['spy_close_price'][:-20]
y_test = stock['spy_close_price'][-20:]
y_pred = linear_regressor.predict(X)
print('R-squared = : %.2f' % r2_score(y_test, y_pred))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

