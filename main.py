# coding=utf-8
import split_icons as splitter
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

import os

variant = 10 + (11)
numbers = [0, 2, 4]
layers = range(3, 6)
# activations = ["tanh", "relu"]
activations = ['tanh', 'relu', 'identity', 'logistic']
solvers = ['adam', 'lbfgs', 'sgd']


def getTrainFileName(index):
    return "./rgr/data/MNIST/train/mnist_train{}.jpg".format(index)


def getTestFileName(index):
    return "./rgr/data/MNIST/test/mnist_test{}.jpg".format(index)


np.random.seed(21296)

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


def main():
    # trainX, trainY, testX, testY = data()
    # print "Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape

    xTrain = np.array([])
    yTrain = np.array([])
    yTest = np.array([])
    for number in numbers:
        print "Generate for number=", number
        filesTrain = splitter.split(getTrainFileName(number),
                                    "./rgr/data/split/train/{}".format(number),
                                    width=28, height=28)
        filesTest = splitter.split(getTestFileName(number),
                                   "./rgr/data/split/test/{}".format(number),
                                   width=28, height=28)

        xTrain.put(number, filesTrain)
        yTest.put(number, filesTest)
        yTrain.put(number, number)

    classifier = MLPClassifier(
        activation=activations[0],
        solver=solvers[0],
        hidden_layer_sizes=[3, 6],
        random_state=1
    )
    classifier.fit(xTrain, yTrain)
    result = classifier.predict(yTest)

    print metrics.accuracy_score(yTest, result)
    return


if __name__ == '__main__':
    main()
