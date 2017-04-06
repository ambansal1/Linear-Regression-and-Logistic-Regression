import os
import math
import csv
import numpy as np


def LinReg_ReadInputs(filepath):
    XTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_XTrain.csv'), delimiter=',')
    yTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_yTrain.csv'), delimiter=',')
    XTest = np.genfromtxt(os.path.join(filepath, 'LinReg_XTest.csv'), delimiter=',')
    yTest = np.genfromtxt(os.path.join(filepath, 'LinReg_yTest.csv'), delimiter=',')
    v = XTest.shape[1]

    XTest = (XTest - XTrain.min(axis=0))/ (XTrain.max(axis=0) - XTrain.min(axis = 0))

    XTrain = (XTrain - XTrain.min(axis=0)) / (XTrain.max(axis=0) - XTrain.min(axis=0))




    b = np.ones((XTrain.shape[0], XTrain.shape[1] + 1))
    b[:, 1:b.shape[1]] = XTrain
    XTrain = b

    c = np.ones((XTest.shape[0], XTest.shape[1]+1))
    c[:, 1:c.shape[1]] = XTest
    XTest = c

    return XTrain,yTrain, XTest, yTest


def LinReg_CalcObj(X, y, w):

    b = (np.dot(X,w))

    c = (np.multiply((b-y),(b-y)))

    c = c/(y.shape[0])

    lossVal = c.sum()

    return lossVal


def LinReg_CalcSG(x, y, w):
    sg = np.zeros((w.shape[0],1))
    a = np.dot(x, w)

    b = 2*((a-y))

    sg = np.multiply(x,b).T


    return sg


def LinReg_UpdateParams(w, sg, eta):


    w = w - eta*sg


    return w


def LinReg_SGD(XTrain, yTrain, XTest, yTest):


    trainLoss = []
    testLoss = []

    w = np.zeros((XTrain.shape[1]))

    test = []
    w = w +0.5

    counter = 1
    for number in range(0,175):


        for i in range(0,XTrain.shape[0]):


            sg = LinReg_CalcSG(XTrain[i,:], yTrain[i], w)

            eta = counter
            eta = math.sqrt(eta)
            eta = 0.5 / (eta)

            w = LinReg_UpdateParams(w, sg, eta)
            counter = counter +1

        trainLosse = LinReg_CalcObj(XTrain, yTrain, w)
        testLosse = LinReg_CalcObj(XTest, yTest, w)
        trainLoss.append(trainLosse)
        testLoss.append(testLosse)
        if number >100:
            test1 = LinReg_CalcObj(XTest, yTest, w)
            test.append(test1)

    return (w, trainLoss, testLoss,test)


def plot():  # This function's results should be returned via gradescope and will not be evaluated in autolab.


    return None
