import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
    log_product = 0

    log_product = np.sum(x)

    return log_product

def NB_XGivenY(XTrain, Ytrain, alpha, beta):
    onionind  = np.where(Ytrain == 1)[0]
    economicind = np.where( Ytrain == 0)[0]
    counteco = len(economicind)
    countonion = len(onionind)

    D = np.zeros([2, XTrain.shape[1]])

   # for item in np.nditer(onionind):
    #    for i in range(0,XTrain.shape[1]):
    #D[1,:] = np.sum(XTrain[[onionind[:]],:],axis=0)
    XY = XTrain[[onionind[:]],:]
    D[1,:] = np.sum(XY, axis=1)

    XYZ = XTrain[[economicind[:]],:]
    D[0,:] = np.sum(XYZ, axis=1)

    #XY =  XTrain[[economicind[:]],:]
    #D[0,:] = XY.sum(axis =0)
     #       if XTrain[item][i] == 1:
     #           D[1,i] =+ 1

    #D[0,:] = np.add(XTrain[economicind[:],:],axis =0)

    #for item in np.nditer(economicind):
     #   for i in range(0,XTrain.shape[1]):
      #         if XTrain[item][i] == 1:
       #            D[0,i] =+ 1






    D[0,:] = (D[0,:]+alpha -1 )/(counteco+alpha+beta-2)
    D[1, :] = (D[1, :] + alpha - 1) / (countonion + alpha + beta - 2)



# Divide each member of D with a-1 and b-1 and lenth of onionind and econominind



        # than for each onionind and each word of vocab find out the frequency, INCLUDE ALPHA ANDBEETA
     #   similar for economoin and put them in D

    return D

def NB_YPrior(yTrain):

    itemindex = np.where(yTrain == 0)[0]
    p = float(len(itemindex))

    p = ( p/len(yTrain))
    return p

def NB_Classify(D, p, XTest):

    Likelihoodeconomic =np.zeros(XTest.shape[0])
    Likelihoodonio = np.zeros(XTest.shape[0])
    i = 0
    for j in range(0,XTest.shape[0]):
        value = XTest[j,:]
        W = D[:,:].T
        X = 1-W
        value1 = value.T

        Y = np.log(W[:,0]*value1 + X[:,0]*(1-value1))
        Z = np.log(W[:,1]*value1 + X[:,1]*(1-value1))
        Likelihoodeconomic[i] = logProd(Y) + np.log(p)
        Likelihoodonio[i] = logProd(Z) + np.log(1-p)
        i = i +1
    yHat = np.ones(XTest.shape[0])
    for i in range(0,XTest.shape[0]):
        if Likelihoodeconomic[i] > Likelihoodonio[i]:
            yHat[i] = 0
    return yHat



def classificationError(yHat, yTruth):

    counterror = 0
    for i in range(0,len(yHat)):

        if yHat[i] == yTruth[i]:
            counterror += 1

    error1 = float(len(yTruth))


    error = float(counterror/error1)
    error = 1- error

    return error

