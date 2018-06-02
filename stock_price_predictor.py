import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt 

dates = []
prices = []

#TODO: incorporate pandas instead.
def get_data(AMZN.csv):
	with open(AMZN.csv, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)

		next(csvFileReader)
		for row in csvFileReader:

			dates.append(int(row[0].split('')[0]))
			prices.append(float(row[1]))
	return


def predict_prices(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1))

	svr_lin = SVR(kernel = 'linear', C = 1e3)
	svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
	svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
	svr_lin.fit(dtes, prices)
#finish off here

