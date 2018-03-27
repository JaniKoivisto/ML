import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
epsilon = 1e-7

def loadData(filename):
    data = np.genfromtxt(filename, delimiter=',')
    X = data[:,:2]
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    Y = data[:,2].reshape(X.shape[0], 1)
    return X, Y

X, y = loadData("training2.csv")

def run():
    weight, costs = run_gradient_descent(X, y, epsilon, learning_rate)
    print("Cost function value: {}".format(costs[-1]))
    print("Weight vector: \n {}".format(weight))

    plt.plot(costs)
    plt.xlabel("# of iteration")
    plt.ylabel("cost function value")
    plt.show()

def run_gradient_descent(X, y, epsilon, learning_rate):
    w = np.zeros((X.shape[1], y.shape[1]))
    training_cost = -np.inf
    previous_cost_value = np.inf
    costs = []
    
    while np.abs(training_cost - previous_cost_value) >= epsilon:
        grad_value = gradient(X, y, w)
        previous_cost_value = training_cost
        training_cost = cost(X, y, w)
        costs.append(training_cost)
        w = w - learning_rate * grad_value

    return w, costs

def gradient(X, y, w):
    G = (2/X.shape[0]) * np.dot(X.T, np.dot(X, w) - y)
    return np.array(G)

def cost(X, y, w):
    cost = np.mean(np.square(np.dot(X,w) - y))
    return float(cost)

if __name__ == "__main__":
    run()