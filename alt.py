# coding=utf-8
import cv2
import numpy as np
from math import sqrt
from math import ceil
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


def load(files, digits):
    images = []
    Y = []
    for i in range(0, 9):
        img = cv2.imread(files[i], 0)
        height, width = img.shape[:2]
        img = img[0:height, 0:width - s]
        height, width = img.shape[:2]
        for j in range(0, width, s):
            for k in range(0, height, s):
                crop_img = img[k:k + s, j:j + s]
                images.append(crop_img)
                Y.extend([digits[i]])
    return images, Y


def prepare(images, features_extraction):
    features = []
    for image in images:
        features.append(features_extraction(image, s, k))
    return features


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


def calculate(X_train, Y_train, X_test, act_function, layer):
    clf = MLPClassifier(solver='lbfgs',
                        activation=act_function,
                        hidden_layer_sizes=layer,
                        random_state=1)
    clf.fit(X_train, Y_train)
    res = clf.predict(X_test)
    print act_function, "\t", layer, "\t", metrics.accuracy_score(Y_test, res)


s = 28
k = 7
functions = ["identity", "logistic", "tanh", "relu"]
methods = [zones, projections, proj_hist]
train, Y_train = load(["train/mnist_train0.jpg",
                       "train/mnist_train1.jpg",
                       "train/mnist_train2.jpg",
                       "train/mnist_train3.jpg",
                       "train/mnist_train4.jpg",
                       "train/mnist_train5.jpg",
                       "train/mnist_train6.jpg",
                       "train/mnist_train7.jpg",
                       "train/mnist_train8.jpg",
                       "train/mnist_train9.jpg"],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test, Y_test = load(["test/mnist_test0.jpg",
                     "test/mnist_test1.jpg",
                     "test/mnist_test2.jpg",
                     "test/mnist_test3.jpg",
                     "test/mnist_test4.jpg",
                     "test/mnist_test5.jpg",
                     "test/mnist_test6.jpg",
                     "test/mnist_test7.jpg",
                     "test/mnist_test8.jpg",
                     "test/mnist_test9.jpg"],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

nph = int(ceil(sqrt(s * 2 * 10)))  # количество нейронов для метода проекционных гистограмм
npj = int(ceil(sqrt(s * 10)))  # количество нейронов для метода проекций
nz = int(ceil(sqrt((s / k) * (s / k) * 10)))  # количество нейронов для метода зон

layers_hist = [
    (nph, nph, nph, nph),
    (nph, nph, nph, nph, nph),
    (nph, nph, nph, nph, nph, nph),
    (nph, nph, nph, nph, nph, nph, nph)
]

layers_proj = [
    (npj, npj, npj, npj),
    (npj, npj, npj, npj, npj),
    (npj, npj, npj, npj, npj, npj),
    (npj, npj, npj, npj, npj, npj, npj)
]

layers_zones = [
    (nz, nz, nz, nz),
    (nz, nz, nz, nz, nz),
    (nz, nz, nz, nz, nz, nz),
    (nz, nz, nz, nz, nz, nz, nz)
]
layers = {zones: layers_zones, projections: layers_proj, proj_hist: layers_hist}

for method in methods:
    X_train = prepare(train, method)
    X_test = prepare(test, method)
    print method
    layers1 = layers[method]
    for layer in layers1:
        for function in functions:
            calculate(X_train, Y_train, X_test, function, layer)
