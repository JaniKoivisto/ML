import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

learning_rate = 0.01
epsilon = epsilon=1e-7

data = loadData("training2.csv")
train_x = data[:,:2]
N = train_x.shape[0]
train_x = np.hstack((np.ones((train_x.shape[0], 1)), train_x))

train_y = data[:, 2].reshape(N,1)

X = tf.placeholder(tf.float32, [N, train_x.shape[1]])
Y = tf.placeholder(tf.float32, [N, 1])

weigth = tf.Variable(tf.zeros([train_x.shape[1],train_y.shape[1]]), name="w")
bias = tf.Variable(tf.zeros([1]), name="b")

with tf.name_scope("linear_model") as scope:
    # W * X + b
    model = tf.add(tf.matmul(X, weigth), bias)

with tf.name_scope("cost") as scope:
    # MSE
    cost = tf.reduce_mean(tf.squared_difference(model,Y))

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_cost = -np.inf
    previous_cost_value = np.inf
    costs = []
    iteration = 0
    while np.abs(training_cost - previous_cost_value) >= epsilon:
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer, {X: train_x, Y: train_y})
            previous_cost_value = training_cost
            training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            costs.append(training_cost)
            if (iteration + 1) % 100 == 0:
                print("Iteration: ", (iteration + 1), "Training cost:", previous_cost_value, "Weigth=", sess.run(weigth))
            iteration += 1
    print("Training completed!")
    print("Iterations: ", (iteration), "Training cost:", costs[-1], "Weigth=", sess.run(weigth))
    
    plt.plot(costs)
    plt.xlabel("# of iteration")
    plt.ylabel("cost function value")
    plt.show()
