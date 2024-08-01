import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

global iterations
global progress
global completed
completed = False
progress = 0
iterations = 500

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape # m = amt rows, n = amt features + 1 (since labeled)
np.random.shuffle(data)

data_dev = data[0:1000].T # .T to transpose... each column is example instead of row
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

#print(Y_train)

#print(X_train[:, 0].shape) # expected output: 784... we should have 784 pixels in this column

def init_params():
    W1 = np.random.rand(10, 784) - 0.5 # generates random values between 0 and 1 for each pixel in array -> subtract 0.5 for -0.5 to 0.5 range
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    """Linear function returning X if x > 0 or 0 if x <= 0"""
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propogation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    """encode Y labels and turn them into a matrix"""
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # matrix of 0's matching n (# examples)
    one_hot_Y[np.arange(Y.size), Y] = 1 # np.arange(Y.size) creates range 0, n (# examples)... Y specifies which column is being set to 1
    one_hot_Y = one_hot_Y.T # transpose to flip so each column is an example
    return one_hot_Y

def derivative_of_ReLU(Z):
    return Z > 0 # if Z > 0, continuous slope of 1, anything less than 0, horizontal line, trend/slope of 0

def backward_propogation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derivative_of_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    global progress
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propogation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propogation(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, Y))
            progress += 10
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propogation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def predict_user_input(input, W1, b1, W2, b2):
    prediction = make_predictions(input, W1, b1, W2, b2)
    return prediction

def train():
    global completed
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations, 0.1)
    completed = True


if __name__ == "__main__":
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)

    test_prediction(124, W1, b1, W2, b2)
    test_prediction(1450, W1, b1, W2, b2)
    test_prediction(1761, W1, b1, W2, b2)
    test_prediction(10233, W1, b1, W2, b2)
    test_prediction(1588, W1, b1, W2, b2)
    test_prediction(8333, W1, b1, W2, b2)
    test_prediction(26443, W1, b1, W2, b2)
    test_prediction(12853, W1, b1, W2, b2)
    test_prediction(36320, W1, b1, W2, b2)