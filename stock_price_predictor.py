
import matplotlib.pyplot as plt 
from matplotlib import style
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import csv

style.use('ggplot')

#read csv file using pandas 
df = pd.read_csv('/Users/hernanrazo/pythonProjects/stock_price_predictor/AMZN.csv', engine = 'python')

#convert date column to type datetime64 
df['Date'] = pd.to_datetime(df.Date)

#return only day numbers
dates = df.Date.dt.day

#return only closing price values
closing_price = df['Close'].values


def predict_price(dates, closing_price, x):
	
	#reshape dates array 
	dates = np.reshape(dates,(len(dates), 1))

	#create models
	svr_lin = SVR(kernel = 'linear', C = 1e3)
	svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
	svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
	
	#train models
	svr_lin.fit(dates, closing_price)
	svr_poly.fit(dates, closing_price)
	svr_rbf.fit(dates, closing_price)

	plt.scatter(dates, prices, color= 'black', label = 'Data')
	plt.plot(dates, svr_lin.predict(dates), color = 'green', label = 'Linear model')
	plt.plot(dates, svr_poly.predict(dates), color = 'blue', label = 'Polynomial model')
	plt.plot(dates, svr_rbf.predict(dates), color = 'red', label = 'RBF model')

	#format labels and overall appearance of graph
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('AMZN Stock Prediction')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

predicted_price = predict_price(dates, closing_price, 29)
print(predicted_price)


