import os
import math
import numpy as np


def LogReg_ReadInputs(filepath):

    XTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_XTrain.csv'), delimiter=',')
    yTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_yTrain.csv'), delimiter=',')
    XTest = np.genfromtxt(os.path.join(filepath, 'LogReg_XTest.csv'), delimiter=',')
    yTest = np.genfromtxt(os.path.join(filepath, 'LogReg_yTest.csv'), delimiter=',')

    b = np.ones((XTrain.shape[0], XTrain.shape[1] + 1))
    b[:, 1:b.shape[1]] = XTrain
    XTrain = b

    c = np.ones((XTest.shape[0], XTest.shape[1] + 1))
    c[:, 1:c.shape[1]] = XTest
    XTest = c

    return (XTrain, yTrain, XTest, yTest)
    
def LogReg_CalcObj(X, y, w):


    a = np.dot(X, w)
    a = 1 /(1+ np.exp(-1*a))

    f= ((1-y)*np.log(1 - a)) + (y*np.log(a))

    lossVal = (np.sum(f))/y.shape[0]


    #function that outputs the conditional log likelihood we want to maximize.

    #Input
    #w      : numpy weight vector of appropriate dimensions initialized to 0.5
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #cll   : The conditional log likelihood we want to maximize
    
    cll = lossVal
    
    return cll
    
def LogReg_CalcSG(x, y, w):



    a = np.dot(x, w)


    a = 1 /(1+ np.exp(-a))

    b = ((y-a))

    sg = x*b



    return sg

        
def LogReg_UpdateParams(w, sg, eta):
    

    w = w + sg*eta


    
    return w
    
def LogReg_PredictLabels(X, y, w):
    a = np.dot(X, w)
    a = 1/(1+np.exp(-a))
    indices = np.where(a > 0.5)[0]

    yPred = np.zeros(y.shape[0])
    yPred[indices] = 1
    counterror = 0

    for i in range(0,y.shape[0]):

        if yPred[i] != y[i]:
            counterror += 1


    error1 = float(len(y))

    error = float(counterror / error1)

    PerMiscl = error


    return (yPred, PerMiscl)

def LogReg_SGA(XTrain, yTrain, XTest, yTest):
    

    trainPerMiscl = []
    testPerMiscl = []

    w = np.zeros((XTrain.shape[1]))

    w = w + 0.5

    counter = 1

    for number in range(0, 5):
        print XTest.shape[0]

        for i in range(0, XTrain.shape[0]):
            sg = LogReg_CalcSG(XTrain[i, :], yTrain[i], w)

            eta = 0.5 / (math.sqrt(counter))

            w = LogReg_UpdateParams(w, sg, eta)





            if  counter%200 == 0:
                _,trainPerMiscle = LogReg_PredictLabels(XTrain, yTrain, w)
                P,testPerMiscle = LogReg_PredictLabels(XTest, yTest, w)


                trainPerMiscl.append(trainPerMiscle)
                testPerMiscl.append(testPerMiscle)
            counter = counter + 1

    yPred, Perm = LogReg_PredictLabels(XTest, yTest, w)

    

    
    return (w, trainPerMiscl, testPerMiscl, yPred, Perm)

