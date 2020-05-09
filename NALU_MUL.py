def Shaurya(X, shape):
    Y = np.zeros(shape)  # Function to multiply elements in the list
    for i in range(X.shape[0]):
        x = X[i]
        x = np.prod(x)
        Y[i] = int(x)
    return Y


import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from numpy import loadtxt

# Loading training and testing set and deleting rows to create data set for list sizes 2, 3 and 4
data6 = loadtxt('data6.csv', delimiter=',')
data2 = np.delete(data6, (2, 3, 4, 5), 1)
data8 = loadtxt('data8.csv', delimiter=',')
data3 = np.delete(data8, (3, 4, 5, 6, 7), 1)
data10 = loadtxt('data10.csv', delimiter=',')
data4 = np.delete(data10, (4, 5, 6, 7, 8, 9), 1)

# Forming training and testing set for list sizes 2, 3 and 4
# In range is testing set in same range as training set
# Out range is testing set in a different range from training set
X_train6 = data2[0:50000, :]
X_test6_inrange = data2[50000:51000, :]
X_test6_outrange = data2[51000:52000, :]

X_train8 = data3[0:50000, :]
X_test8_inrange = data3[50000:51000, :]
X_test8_outrange = data3[51000:52000, :]

X_train10 = data4[0:50000, :]
X_test10_inrange = data4[50000:51000, :]
X_test10_outrange = data4[51000:52000, :]

array_lengths = [2, 3, 4]
epsilon_values = [1e-7, 1e-6, 1e-5]

for al in array_lengths:

    if (al == 2):
        X_train = X_train6
        Y_train = Shaurya(X_train, (50000, 1))
        X_test1 = X_test6_inrange
        Y_test1 = Shaurya(X_test1, (1000, 1))
        X_test2 = X_test6_outrange
        Y_test2 = Shaurya(X_test2, (1000, 1))
    elif (al == 3):
        X_train = X_train8
        Y_train = Shaurya(X_train, (50000, 1))
        X_test1 = X_test8_inrange
        Y_test1 = Shaurya(X_test1, (1000, 1))
        X_test2 = X_test8_outrange
        Y_test2 = Shaurya(X_test2, (1000, 1))
    elif (al == 4):
        X_train = X_train10
        Y_train = Shaurya(X_train, (50000, 1))
        X_test1 = X_test10_inrange
        Y_test1 = Shaurya(X_test1, (1000, 1))
        X_test2 = X_test10_outrange
        Y_test2 = Shaurya(X_test2, (1000, 1))

    for ev in epsilon_values:

        print("This section is for a list of length", al, "and epsilon as", ev)
        X = tf.placeholder(np.float32, shape=[None, al])
        Y = tf.placeholder(np.float32, shape=[None, 1])

        #  Defining Neural Arithmetic Logic unit
        w_hat = tf.Variable(tf.truncated_normal([al, 1], stddev=0.02))
        m_hat = tf.Variable(tf.truncated_normal([al, 1], stddev=0.02))
        G = tf.Variable(tf.truncated_normal([al, 1], stddev=0.02))

        W = tf.tanh(w_hat) * tf.sigmoid(m_hat)
        a = tf.matmul(X, W)
        g = tf.sigmoid(tf.matmul(X, G))
        m = tf.exp(tf.matmul(tf.log(tf.abs(X) + ev), W))
        output = g * a + (1 - g) * m

        cost = tf.losses.mean_squared_error(labels=Y, predictions=output)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            cost_list = []
            epoch_list = []
            cost_list1 = []
            cost_list2 = []

            # Training
            for epoch in range(20):
                batches = 0
                while batches < len(X_train):
                    X_training_batches = X_train[batches:batches + 16, :]
                    Y_train_batches = Shaurya(X_training_batches, (16, 1))
                    _, c = sess.run([optimizer, cost], feed_dict={X: X_training_batches, Y: Y_train_batches})
                    batches += 16
                print("Epoch:", (epoch + 1), "training cost =", "{:.15f}".format(c))
                epoch_list.append(epoch + 1)
                cost_list.append(c)
                c1 = sess.run((cost), feed_dict={X: X_test1, Y: Y_test1})
                cost_list1.append(c1)
                c2 = sess.run((cost), feed_dict={X: X_test2, Y: Y_test2})
                cost_list2.append(c2)

            # Plotting outputs
            # plt.plot(epoch_list, cost_list)
            # plt.plot(epoch_list, cost_list1)
            # plt.plot(epoch_list, cost_list2)
            # plt.xlabel('Iteration')
            # plt.ylabel('Cost')
            # plt.show()

            # Finding accuracy and cost for 1st testing set
            predictions = []
            predictions = np.round(sess.run(output, feed_dict={X: X_test1}))
            count = 0
            for i in range(1000):
                if predictions[i] == Y_test1[i]:
                    count += 1
            print("inrange accuracy", ":", count / 10, '%')

            predictions = np.asarray(predictions).reshape(1000, 1)
            testing_cost_inrange = (1 / 50000) * np.dot((predictions - Y_test1).T, (predictions - Y_test1))
            print("The cost for testing set of the same range as the training set:", float(testing_cost_inrange))

            # Finding accuracy and cost for 2nd testing set
            prediction = []
            predictions = np.round(sess.run(output, feed_dict={X: X_test2}))
            count = 0
            for i in range(1000):
                if predictions[i] == Y_test2[i]:
                    count += 1
            print("outrange accuracy", ":", count / 10, '%')

            predictions = np.asarray(predictions).reshape(1000, 1)
            testing_cost_outrange = (1 / 50000) * np.dot((predictions - Y_test2).T, (predictions - Y_test2))
            print("The cost for testing set in a different range from the training set", float(testing_cost_outrange))
        sess.close()
