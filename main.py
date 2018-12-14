import split_icons as splitter
import os

def getTraintFileName(index):
    return "./rgr/data/MNIST/train/mnist_train{}.jpg".format(index)

def getTestFileName(index):
    return "./rgr/data/MNIST/test/mnist_test{}.jpg".format(index)

def main():
    # trainX, trainY, testX, testY = data()
    # print "Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape

    splitter.use(getTraintFileName(0), "./rgr/data/form/", 28, 28)

    return


if __name__ == '__main__':
    main()
