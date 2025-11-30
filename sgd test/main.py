import numpy as np
import plotly.express as px
from numba import jit


@jit(nopython=True)  # add numba
def predict(theta, xs):
    return np.dot(xs, theta)


@jit(nopython=True)
def J_squared_residual(theta, xs, y):
    h = predict(theta, xs)
    sr = ((h - y) ** 2).sum()
    return sr


@jit(nopython=True)
def gradient_J_squared_residual(theta, xs, y):
    h = predict(theta, xs)
    grad = np.dot(xs.transpose(), (h - y))
    return grad


# the dataset (already augmented so that we get an intercept coef)
# remember: augmented x -> we add a colum of 1's instead of using a bias term.
data_x = np.array([[1.0, 0.5], [1.0, 1.0], [1.0, 2.0]])
data_y = np.array([[1.0], [1.5], [2.5]])
n_features = data_x.shape[1]

# variables we need
theta = np.zeros((n_features, 1))
learning_rate = 0.1
m = data_x.shape[0]
batch_size = 2          # Antall batches
n_iters = 10

# run GD
j_history = []

@jit(nopython=True)
# Mini-batch SGD Function
def sgd_minibatch(theta, data_x, data_y, batch_size, learning_rate, n_iters, m):
    j_history = np.zeros(n_iters)
    for it in range(n_iters):
        # Shuffle data at the beginning of each iteration
        indices = np.random.permutation(m)
        data_x_shuffled = data_x[indices]
        data_y_shuffled = data_y[indices]

        for i in range(0, m, batch_size):
            x_batch = data_x_shuffled[i:i + batch_size]
            y_batch = data_y_shuffled[i:i + batch_size]

            # Update theta for each mini-batch
            #theta = theta - (learning_rate * (1 / batch_size) * gradient_J_squared_residual(theta, x_batch, y_batch))
            # Bytter til dette:
            theta -= learning_rate * gradient_J_squared_residual(theta, x_batch, y_batch) / batch_size



    # Calculate and store the loss after processing all mini-batches
    j = J_squared_residual(theta, data_x, data_y)
#    j_history.append(j)

    # Track loss after all mini-batches
    j_history[it] = J_squared_residual(theta, data_x, data_y)
    return theta, j_history

# Kjører min-batch SGD
theta, j_history = sgd_minibatch(theta, data_x, data_y, batch_size, learning_rate, n_iters, m)


print("theta shape:", theta.shape)

# append the final result.
#j = J_squared_residual(theta, data_x, data_y)
#j_history.append(j)
#print("The L2 error is: {:.2f}".format(j))

# find the L1 error.
y_pred = predict(theta, data_x)
l1_error = np.abs(y_pred - data_y).sum()
print("The L1 error is: {:.2f}".format(l1_error))

# Find the R^2
# if the data is normalized: use the normalized data not the original data (task 3 hint).
# https://en.wikipedia.org/wiki/Coefficient_of_determination

# R^2 calculation
u = ((data_y - y_pred) ** 2).sum()
v = ((data_y - data_y.mean()) ** 2).sum()

print("R^2: {:.2f}".format(1 - (u / v)))

# plot the result
fig = px.line(j_history, title="J(theta) - Loss History")
fig.show()



