import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(filename):
	df = pd.read_csv(filename, sep=",", index_col=False)
	df.columns = ["test-1", "test-2", "result"]
	data = np.array(df, dtype=float)
	plot_data(data[:,:2], data[:, -1])
	return data[:,:2], data[:, -1]

def plot_data(x, y):
	plt.xlabel('score of test-1')
	plt.ylabel('score of test-1')
	for i in range(x.shape[0]):
		if y[i] == 1:
			plt.plot(x[i,0], x[i,1], 'gX')
		else:
			plt.plot(x[i,0], x[i,1], 'mD')
	plt.show()

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def cost_function(x, y, theta):
	h = sigmoid(x@theta)
	print(np.log(h).shape)
	one = np.ones((y.shape[0],1))
	return (-((y.T@np.log(h)) + (one-y).T@np.log(one - h))/(y.shape[0]))

def gradient_descent(x, y, theta, learning_rate=0.1, num_epochs=10):
	m = x.shape[0]
	J_all = []
	
	for _ in range(num_epochs):
		h_x = sigmoid(x@theta)
		cost_ = (1/m)*(x.T@(h_x - y))
		theta = theta - (learning_rate)*cost_
		J_all.append(cost_function(x, y, theta))

	return theta, J_all 


def plot_cost(J_all, num_epochs):
	plt.xlabel('Epochs')
	plt.ylabel('Cost')
	plt.plot(num_epochs, J_all, 'm', linewidth = "5")
	plt.show()

def predict(prob):
	if(prob >= 0.5):
		return 1
	else:
		return 0

def test(theta, x):
	y = float(sigmoid(x@theta))
	if predict(y) == 1 :
		print("Admit")
	else:
		print("Reject")



x, y = load_data("student_result_data.txt")
y = np.reshape(y, (y.shape[0], 1))
x = np.hstack((np.ones((x.shape[0], 1)), x))
theta = np.zeros((x.shape[1], 1))
learning_rate = 0.001
num_epochs = 100
theta, J_all = gradient_descent(x, y, theta, learning_rate, num_epochs)
J = cost_function(x, y, theta)
print(theta)
print(J)

n_epochs = []
jplot = []
count = 0
for i in J_all:
	jplot.append(i[0][0])
	n_epochs.append(count)
	count += 1
jplot = np.array(jplot)
n_epochs = np.array(n_epochs)
plot_cost(jplot, n_epochs)

test(theta, [1, 48, 85])
