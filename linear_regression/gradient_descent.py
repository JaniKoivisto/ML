import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

def readData(filename):
    xs = []
    ys = []
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            xs.append(row[0])
            ys.append(row[1])
 
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

train_x,train_y = readData("train_data.csv")
learning_rate = 0.01
samples = len(train_x)
epsilon = epsilon=1e-7

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

weigth = tf.Variable(0.0)
bias = tf.Variable(0.0)

with tf.name_scope("linear_model") as scope:
    # W * X + b
    model = tf.add(tf.multiply(weigth, X), bias)

with tf.name_scope("cost") as scope:
    # MSE
    cost = tf.reduce_sum(tf.pow(model-Y, 2))/(2 * samples)

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_cost = -np.inf
    previous_cost_value = np.inf
    iteration = 0
    while np.abs(training_cost - previous_cost_value) >= epsilon:
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer, {X: x, Y: y})
            training_cost = previous_cost_value
            previous_cost_value = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            if (iteration + 1) % 100 == 0:
                print("Iteration: ", (iteration + 1), "Training cost:", previous_cost_value, "weigth=", sess.run(weigth))
            iteration += 1
    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    print("Training completed!")
    print("Iterations: ", (iteration), "Training cost:", training_cost, "weigth=", sess.run(weigth))

    '''
    plt.plot(train_x, train_y, 'ro', label='Observed data')
    plt.plot(train_x, sess.run(weigth) * train_x + sess.run(b), label='Prediction')
    plt.legend()
    plt.show()

    '''