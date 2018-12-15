# coding=utf-8
import abc

import enum

import split_icons as splitter
import numpy as np
import cv2
from math import sqrt
from math import ceil
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from enum import Enum, unique

import os

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
variant = 10 + (11)
numbers = [0, 2, 4]
layers = range(3, 6)
# activations = ["tanh", "relu"]
activations = ['tanh', 'relu', 'identity', 'logistic']
solvers = ['adam', 'lbfgs', 'sgd']

imageSize = 28
k = 7
imagePixels = imageSize * imageSize


class Source(Enum):
    train = "train"
    test = "test"

    def getFileName(self, type, number):
        return "./rgr/data/MNIST/{0}/mnist_{0}{1}.jpg".format(type, number)

    def getCroppedPath(self, number):
        return "./rgr/data/split/{0}/{1}".format(type, number)


def zones(image, s, k):
    features = []
    w = s / k
    count = [0 for x in range(k * k)]
    for i in range(s):
        for j in range(s):
            if image[i][j] > 200:  # сравнение с пороговым в RGB
                count[i // w * k + j // w] += 1  # число пикселей в зонах
    for l in range(k * k):
        count[l] /= float(w * w)  # для кадоый зоны рассчитывается плотность пикселей
    features.extend(count)
    return features


def projections(image, s, k):
    features = []
    w = s / k
    for i in range(s):
        count = [0 for x in range(s)]
        for j in range(s):
            if image[i][j] > 200:
                count[i] += 1
        features.extend(count)
    return features


def proj_hist(image, s, k):
    features = []
    count = [0 for x in range(s + s)]
    for i in range(s):
        for j in range(s):
            if image[i][j] > 200:
                count[i] += 1
    for i in range(s):
        for j in range(s):
            if image[j][i] > 200:
                count[i * 2] += 1
    features.extend(count)
    return features


def getArray(source):
    features = []
    labels = []

    for number in range(0, 9):
        imageRaw = cv2.imread(Source.getFileName(Source.train, source, number), 0)
        height, width = imageRaw.shape[:2]
        imageRaw = imageRaw[0:height, 0:width - imageSize]
        height, width = imageRaw.shape[:2]
        for j in range(0, width, imageSize):
            for k in range(0, height, imageSize):
                imageCropped = imageRaw[k:k + imageSize, j:j + imageSize]
                features.append(imageCropped)
                labels.extend([number])
    return features, labels


np.random.seed(21296)

train, train_labels = getArray("train")
test, test_labels = getArray("test")

"""
Задача классификации
У нас имеется 10 классов (10 цифр)
Каждое изображение содержит цвет от 0 до 255, имеет размер 28х28

Нейронная сеть имеет 2 уровня: входной и выходной
В качестве входных сигналов используются значения интенсивности цветовых сигналов (28*28=784)
Входной слой будет иметь 800 нейронов
Выходной слой будет 10 нейронов
Каждый нейрон соответствует классу цифр (от 0 до 9)
Выходное значение нейрона соответствует вероятности, что на картинке представлена та или иная цифра
"""


def prepare(images, features_extraction):
    features = []
    for image in images:
        features.append(features_extraction(image, imageSize, k))
    return features


def calculate(X_train, Y_train, X_test, act_function, layer):
    clf = MLPClassifier(solver='lbfgs',
                        activation=act_function,
                        hidden_layer_sizes=layer,
                        random_state=1)
    clf.fit(X_train, Y_train)
    res = clf.predict(X_test)
    print act_function, "\t", layer, "\t", metrics.accuracy_score(test_labels, res)


def main():
    # trainX, trainY, testX, testY = data()
    # print "Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape

    nph = int(ceil(sqrt(imagePixels * 2 * 10)))  # количество нейронов для метода проекционных гистограмм
    npj = int(ceil(sqrt(imagePixels * 10)))  # количество нейронов для метода проекций
    nz = int(ceil(sqrt((imagePixels / k) * (imagePixels / k) * 10)))  # количество нейронов для метода зон

    layers_hist = [
        (nph, nph, nph),
        (nph, nph, nph, nph),
        (nph, nph, nph, nph, nph),
        (nph, nph, nph, nph, nph, nph)
    ]

    layers_proj = [
        (npj, npj, npj),
        (npj, npj, npj, npj),
        (npj, npj, npj, npj, npj),
        (npj, npj, npj, npj, npj, npj)
    ]

    layers_zones = [
        (nz, nz, nz),
        (nz, nz, nz, nz),
        (nz, nz, nz, nz, nz),
        (nz, nz, nz, nz, nz, nz)
    ]
    layers = {zones: layers_zones, projections: layers_proj, proj_hist: layers_hist}
    methods = [zones, projections, proj_hist]

    for method in methods:
        X_train = prepare(train, method)
        X_test = prepare(test, method)
        print method
        layers1 = layers[method]
        for layer in layers1:
            for function in activations:
                calculate(X_train, train_labels, X_test, function, layer)


if __name__ == '__main__':
    main()
