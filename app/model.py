#
# Title: CNN Model for number recognition
# Authors: Andreas Weber
# Created on Fri Oct 22 2021
#
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


class CNN:
    def predict_L_layer(self, X, parameters):
        AL, caches = self.L_model_forward(X, parameters)
        prediction = np.argmax(AL, axis=0)
        return prediction.reshape(1, prediction.shape[0])

    def fit_and_transform(self) -> StandardScaler:
        number_data = load_digits()
        n_samples = len(number_data.images)
        x = number_data.images.reshape((n_samples, -1))
        y = number_data.target

        # split training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0
        )

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)  # fit data, then transform it
        X_test = sc.transform(
            X_test
        )  # already fitted with train data, only transform test data
        return sc

    # use ful activation functions and their derivatives
    def sigmoid_(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu_(self, Z):
        return Z * (Z > 0)

    def drelu_(self, Z):
        return 1.0 * (Z > 0)

    def dsigmoid_(self, Z):
        return self.sigmoid_(Z) * (1 - self.sigmoid_(Z))

    def sigmoid(self, Z):
        return self.sigmoid_(Z), Z

    def relu(self, Z):
        return self.relu_(Z), Z

    def linear_forward(self, A, W, b):
        """Forward propagation without activation for a single layer

        Args:
            A ([type]): [description]
            W ([type]): [description]
            b ([type]): [description]

        Returns:
            [touple]: [description]
        """
        Z = np.dot(W, A) + b
        assert Z.shape == (W.shape[0], A.shape[1])
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """Forward propagation with activation for a single layer:
        If the type of activation is sigmoid, it performs sigmoid activation function else performs relu activation function.

        Args:
            A_prev ([type]): [description]
            W ([type]): [description]
            b ([type]): [description]
            activation ([type]): [description]

        Returns:
            [type]: [description]
        """

        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)

        assert A.shape == (W.shape[0], A_prev.shape[1])
        cache = (linear_cache, activation_cache)

        return A, cache

    # implementation of forward propogation for L layer neural network
    def L_model_forward(self, X, parameters):
        """Output the final predicted vector of the output layer and some cache information

        chache: information which is used for backpropagation

        Args:
            X ([type]): [description]
            parameters ([type]): [description]

        Returns:
            touple: AL, caches
        """
        caches = []
        A = X
        L = len(parameters) // 2
        # For the first L-1 layers, we use relu as activation function and for the last layer, we use sigmoid activation function
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu"
            )
            caches.append(cache)
        AL, cache = self.linear_activation_forward(
            A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid"
        )
        caches.append(cache)
        # assert(AL.shape == (1,X.shape[1]))
        return AL, caches

    # cost function
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))
        cost = np.squeeze(cost)
        assert cost.shape == ()
        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db

    def relu_backward(self, dA, activation_cache):
        return dA * self.drelu_(activation_cache)

    def sigmoid_backward(self, dA, activation_cache):
        return dA * self.dsigmoid_(activation_cache)

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    # back propogation for L layers
    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        # Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        (
            grads["dA" + str(L - 1)],
            grads["dW" + str(L)],
            grads["db" + str(L)],
        ) = self.linear_activation_backward(dAL, current_cache, "sigmoid")

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, "relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads

    # update parameters
    def update_parameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l + 1)] = (
                parameters["W" + str(l + 1)]
                - (learning_rate) * grads["dW" + str(l + 1)]
            )
            parameters["b" + str(l + 1)] = (
                parameters["b" + str(l + 1)]
                - (learning_rate) * grads["db" + str(l + 1)]
            )
        return parameters