# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

NOT_IMPLEMENTED_ERROR = NotImplementedError("Please Implement this method")


class AbstractClassifier:
    """
    Abstract class to represent a classifier
    """

    def __init__(self):
        raise NOT_IMPLEMENTED_ERROR

    def fit(self, X, y):
        """
        Allows to fit the model on the given set

        Arguments :
            - X: ndarray with samples
            - y: ndarray with corresponding labels
        """
        raise NOT_IMPLEMENTED_ERROR

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        raise NOT_IMPLEMENTED_ERROR

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        yhat = self.predict(X)
        return np.where(y == yhat, 1., 0.).mean()


class Perceptron(AbstractClassifier):

    def __init__(self, learning_rate, max_iter=100, threshold=0.01, bias=0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.bias = bias

        self.weights = []
        self.classes = []
        self.dim = 0

    def __predict(self, x):
        return int(np.sign(np.dot(self.weights, x) + self.bias))

    def fit(self, X, y):
        """
        Allows to fit the model on the given set

        Arguments :
            - X: ndarray with samples
            - y: ndarray with corresponding labels
        """

        (nX, d) = X.shape
        (nY, ) = y.shape

        if nX != nY:
            raise ValueError("X and Y must be the same size")

        self.classes = np.unique(y)

        if len(self.classes) != 2:
            raise ValueError("Y must have two classes")

        def refactor_y(y): return -1 if y == self.classes[0] else 1
        refactor_y = np.vectorize(refactor_y)
        y = refactor_y(y)

        self.dim = d
        self.weights = np.zeros(self.dim)
        old_weights = None

        nb_iter = 0
        diff = self.threshold

        while (nb_iter < self.max_iter) and (diff >= self.threshold):

            old_weights = self.weights.copy()

            for i in np.arange(nX):

                if self.__predict(X[i]) == y[i]:
                    continue

                self.weights += self.learning_rate * y[i] * X[i]

            diff = np.linalg.norm(np.abs(self.weights - old_weights))
            nb_iter += 1

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """

        yhat = []

        (nX, dim) = X.shape

        if dim != self.dim:
            raise ValueError(
                "The dimension of the samples must be equal to that of the inputs")

        for i in np.arange(nX):
            yhat.append(self.classes[min(self.__predict(X[i]) + 1, 1)])

        return np.array(yhat)


class KNN(AbstractClassifier):

    def __init__(self, k):
        self.k = k

        self.dim = 0
        self.X = None
        self.y = None

    def __distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def fit(self, X, y):
        """
        Allows to fit the model on the given set

        Arguments :
            - X: ndarray with samples
            - y: ndarray with corresponding labels
        """

        (nX, d) = X.shape
        (nY, ) = y.shape

        if nX != nY:
            raise ValueError("X and Y must be the same size")

        self.X = X
        self.y = y
        self.dim = d

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """

        yhat = []

        for x in X:
            dist_points = np.argsort(np.array([self.__distance(x, p)
                                     for p in self.X]))

            k_points = dist_points[0:self.k]
            yhat.append(np.unique(self.y[k_points])[0])

        return np.array(yhat)


class LineaireRandom(AbstractClassifier):

    def __init__(self):
        self.weights = None
        self.classes = None
        self.dim = 0

    def __predict(self, x):
        return int(np.sign(np.dot(self.weights, x)))

    def fit(self, X, y):
        """
        Allows to fit the model on the given set

        Arguments :
            - X: ndarray with samples
            - y: ndarray with corresponding labels
        """

        (nX, d) = X.shape
        (nY, ) = y.shape

        if nX != nY:
            raise ValueError("X and Y must be the same size")

        self.dim = d

        self.classes = np.unique(y)

        if len(self.classes) != 2:
            raise ValueError("Y must have two classes")

        w = np.random.uniform(-1, 1, d)
        self.weights = w / np.linalg.norm(w)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """

        yhat = []

        (nX, dim) = X.shape

        if dim != self.dim:
            raise ValueError(
                "The dimension of the samples must be equal to that of the inputs")

        for i in np.arange(nX):
            yhat.append(self.classes[min(self.__predict(X[i]) + 1, 1)])

        return np.array(yhat)


class OAA(AbstractClassifier):

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifiers = []
        self.classes = None

    def fit(self, X, y):
        """
        Allows to fit the model on the given set

        Arguments :
            - X: ndarray with samples
            - y: ndarray with corresponding labels
        """

        (nX, _) = X.shape
        (nY, ) = y.shape

        if nX != nY:
            raise ValueError("X and Y must be the same size")

        self.classes = np.unique(y)
        number_classes = len(self.classes)

        self.classifiers = [deepcopy(
            self.base_classifier) for _ in range(number_classes)]

        def refresh_y(y, c): return 1 if y == c else -1
        refresh_y = np.vectorize(refresh_y)

        for i in range(number_classes):
            y_class = refresh_y(y, self.classes[i])
            self.classifiers[i].fit(X, y_class)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        yhat = []

        yhats = []

        for classif in self.classifiers:
            yhats.append(classif.predict(X))

        (nX, _) = X.shape

        for i in np.arange(nX):
            scores = [0 for _ in self.classifiers]

            for j in range(len(self.classifiers)):
                if yhats[j][i] == 1:
                    scores[j] += 1
                else:
                    for k in range(len(self.classifiers)):
                        scores[k] += 1

                    scores[j] -= 1

            p = np.argmax(scores)
            yhat.append(self.classes[p])

        return np.array(yhat)
