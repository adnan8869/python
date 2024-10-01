import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
def load_data(file_path='lab10.txt'):
    """
    Load data from a text file.
    
    Parameters:
    file_path (str): The path to the data file.
    
    Returns:
    tuple: Two numpy arrays, x_train and y_train.
    """
    data = np.loadtxt(file_path, delimiter=',')
    x_train = data[:, 0]
    y_train = data[:, 1]
    return x_train, y_train
# load the dataset
x_train, y_train = load_data()


plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()


# UNQ_C1
# GRADED FUNCTION: compute_cost
def compute_cost(x, y, w, b):
	"""
	Computes the cost function for linear regression.
	Args:
	x (ndarray): Shape (m,) Input to the model (Population of cities)
	y (ndarray): Shape (m,) Label (Actual profits for the cities)
	w, b (scalar): Parameters of the model
	Returns
	total_cost (float): The cost of using w,b as the parameters for linear regression
	to fit the data points in x and y
	"""
	# number of training examples
	m = x.shape[0]
	# You need to return this variable correctly
	total_cost = 0
	### START CODE HERE ###
	yp = np.zeros(m)
	yp = np.dot(x, w) + b
	total_cost = (1 / (2 * m)) * np.sum((yp-y)**2)
	### END CODE HERE ###
	return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
    x (ndarray): Shape (m,) Input to the model (Population of cities)
    y (ndarray): Shape (m,) Label (Actual profits for the cities)
    w, b (scalar): Parameters of the model
    Returns
    dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
    dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """
    # Number of training examples
    m = x.shape[0]
    
    ### START CODE HERE ###
    yp = np.zeros(m)
    yp = np.dot(x, w) + b

    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    dj_db = (1 / m) * np.sum(yp-y)
    dj_dw = (1 / m) * np.sum((yp-y)*x)
    ### END CODE HERE ###
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha
    Args:
    x : (ndarray): Shape (m,)
    y : (ndarray): Shape (m,)
    w_in, b_in : (scalar) Initial values of parameters of the model
    cost_function: function to compute cost
    gradient_function: function to compute the gradient
    alpha : (float) Learning rate
    num_iters : (int) number of iterations to run gradient descent
    Returns
    w : (ndarray): Shape (1,) Updated values of parameters of the model after
    running gradient descent
    b : (scalar) Updated value of parameter of the model after
    running gradient descent
    """
    # number of training examples
    m = len(x)
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in) #avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # Save cost J at each iteration
        if i<100000: # prevent resource exhaustion
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f} ")
    return w, b, J_history, w_history #return w and J,w history for graphing

initial_w = 0.
initial_b = 0.
# some gradient descent settings
iterations = 1500
alpha = 0.01
w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b,
compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)



m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b

# Plot the linear fit
plt.plot(x_train, predicted, c = "b")
# Create a scatter plot of the data.
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()



predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))
predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))