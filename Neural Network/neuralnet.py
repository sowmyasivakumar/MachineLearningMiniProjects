import numpy as np
import csv
import math
import random
import sys
import re
import filecmp
import time

# -------------------------------- inputs to be taken ------------------------------------
#  <train input>
# <validation input> <test input> <dict input> <formatted train out>
# <formatted validation out> <formatted test out> <feature flag>
# <feature dictionary input>.
#
# if __name__ == "__main__":
#
#     trainInputFilename = sys.argv[1]
#     validationInputFilename = sys.argv[2]
#     trainOutputFilename = sys.argv[3]
#     validOutputFilename = sys.argv[4]
#
#     metricsFileName = sys.argv[5]
#     numEpochs = int(sys.argv[6])
#     hiddenUnits = int(sys.argv[7])
#     initFlag = int(sys.argv[8])
#     learningRate = float(sys.argv[9])

#functions
def inputDataFormatting(dataInput):
    # output labels
    Y = dataInput[:, 0]

    # one-hot encoding
    s = np.array([Y])
    Yi = np.zeros((s.size, 4))
    Yi[np.arange(s.size), s.astype('int64')] = 1

    Y = Y.reshape(Y.shape[0], 1)

    # take input from 1st col and fold 1 to the 0th col
    X = np.column_stack((np.ones_like(Y), dataInput[:, 1:]))
    return X,Y,Yi


#function to calculate error rate for both train and test predictions
def errorrate(y_train, y_pred_train,y_test,  y_pred_test):

    TotalNoOfValues = y_test.size
    TestPredErrorCount = 0
    TrainPredErrorCount = 0

    #test error rate
    for x, y in np.nditer([y_test, y_pred_test]):
        if x != y:
            TestPredErrorCount += 1

    TestPredErrorRate = TestPredErrorCount/TotalNoOfValues

    # train error rate

    TotalNoOfValues = y_train.size
    for x, y in np.nditer([y_train, y_pred_train]):
        if x != y:
            TrainPredErrorCount += 1

    TrainPredErrorRate = TrainPredErrorCount / TotalNoOfValues

    return TestPredErrorRate, TrainPredErrorRate


def predict(X, Y, Yi, alpha, beta):

    a = linearForward(X,alpha)
    z_ = 1 / (1 + np.exp(-a))
    z = np.column_stack((np.ones_like(z_[:, 0]), z_))
    b = linearForward(z,beta)
    ypred = np.apply_along_axis(softMaxForward,1,b)
    return ypred

def linearForward(X, theta):
     a = np.dot(X, theta.T)
     return a

def sigmoidForward(a):
        Z_ = 1 / (1 + np.exp(-a))
        Z = np.insert(Z_,0,1)
        return Z

def softMaxForward(b):
    y = np.exp(b)/np.sum(np.exp(b))
    return y

def crossEntropyForward(Yi, ypred):

    crossEntropy = np.sum(-1*Yi*np.log(ypred))
    return crossEntropy


def forward(X,Y,Yi,alpha, beta):

    a = linearForward(X,alpha)
    z = sigmoidForward(a)
    b = linearForward(z,beta)
    ypred = softMaxForward(b)
    J = crossEntropyForward(Yi,ypred)
    return X,a,z,b,ypred,J

def backward(X,Y,Yi,alpha, beta,a, z, b, ypred, J):
    Gj = 1
    Gb = ypred - Yi

    Gbeta = Gb.reshape(Gb.shape[0], 1)*z

    Gz = np.sum((Gb*beta[:,1:].T).T,axis=0)

    Ga = Gz*z[1:]*[1-z[1:]]

    Galpha = Ga.reshape(Ga.shape[1], 1)*X

    return Galpha, Gbeta


def train(trainDataInput,validDataInput, hiddenUnits, initFlag, numEpochs, learningRate):

    MetricsOutString = ''
    EmpiricalString = ''
    X,Y,Yi = inputDataFormatting(trainDataInput)

    # initialize alpha

    if initFlag == 1:
        alpha_ = np.random.uniform(0.0, 1.0, (hiddenUnits, X.shape[1] - 1))
        beta_ = np.random.uniform(0.0, 1.0, (4, hiddenUnits))

    elif initFlag == 2:
        alpha_ = np.zeros((hiddenUnits, X.shape[1] - 1))
        beta_ = np.zeros((4, hiddenUnits))

    alpha = np.column_stack((np.zeros_like(alpha_[:, 0]), alpha_))
    beta = np.column_stack((np.zeros_like(beta_[:, 0]), beta_))

    print('Initial alpha',alpha)
    print('Initial beta', beta)

    # initialise adagrad update params before each epoch
    Salpha = np.zeros_like(alpha)
    Sbeta = np.zeros_like(beta)
    epsilon = 0.00001  # 1e-5
    learnRateAlpha = np.full_like(alpha, learningRate)
    learnRateBeta = np.full_like(beta, learningRate)

    for t in range(0,numEpochs):

        for i in range(0, len(X)):

            # forward pass
            x, a, z, b, ypred, crossentropy = forward(X[i], Y[i], Yi[i], alpha, beta)

            #backward prop
            Galpha, Gbeta = backward(X[i], Y[i], Yi[i], alpha, beta, a, z, b, ypred, crossentropy)

            #Agadrad update for alpha
            Salpha = Salpha + Galpha*Galpha
            stepSizeAlpha = (learnRateAlpha/np.sqrt(Salpha + epsilon))
            alpha = alpha - stepSizeAlpha*Galpha

            print('alpha:',alpha)

            # Agadrad update for beta
            Sbeta = Sbeta + Gbeta * Gbeta
            stepSizeBeta = (learnRateBeta / np.sqrt(Sbeta + epsilon))
            beta = beta - stepSizeBeta * Gbeta

            print('beta:',beta)

        #Calculating avg cross entropy for train

        ypred = predict(X, Y, Yi, alpha, beta)
        Javg = np.average(np.sum(-1*Yi*np.log(ypred), axis=1))
        MetricsOutString = MetricsOutString + str(f'epoch={t+1} crossentropy(train): {Javg}\n')
        print(str(f'epoch={t+1} crossentropy(train):{Javg}'))

        #calculating avg cross entropy for validation

        Xval, Yval, Yival = inputDataFormatting(validDataInput)
        ypredval = predict(Xval, Yval, Yival, alpha, beta)

        Javgval = np.average(np.sum(-1 * Yival * np.log(ypredval), axis=1))
        print(str(f'epoch={t+1} crossentropy(validation):{Javgval}'))
        MetricsOutString = MetricsOutString + str(f'epoch={t+1} crossentropy(validation): {Javgval}\n')
        # MetricsOutString = MetricsOutString + str(Javgval) + '\n'
        # EmpiricalString = EmpiricalString + str(Javg) + '\n'

    YtrainOutput =  (np.argmax(ypred, axis=1)).reshape(Y.shape)
    YvalidOutput = (np.argmax(ypredval, axis=1)).reshape(Yval.shape)

    # error rate calculation
    TestErrorRate, TrainErrorRate = errorrate(Y, YtrainOutput, Yval, YvalidOutput)
    MetricsOutString= MetricsOutString + str(f'error(train): {TrainErrorRate}\n')
    MetricsOutString= MetricsOutString + str(f'error(train): {TestErrorRate}\n')
    print(str(f'error(train): {TrainErrorRate}'))
    print(str(f'error(validation): {TestErrorRate}'))

    return MetricsOutString, YtrainOutput, YvalidOutput



#
# initializing manually

trainInputFilename = 'small_train.csv'
validationInputFilename = 'small_val.csv'
trainOutputFilename = 'smallTrain_out.labels'
validOutputFilename = 'smallValidation_out.labels'
metricsFileName = 'smallMetrics_out.txt'
numEpochs = 1
hiddenUnits =4
initFlag = 2
learningRate = 0.1

# load the input data into numpy array
start = time.time()

trainDataInput = np.genfromtxt(trainInputFilename, delimiter=",")
validDataInput = np.genfromtxt(validationInputFilename, delimiter=",")

MetricsOutString, YtrainOutput, YvalidOutput = train(trainDataInput, validDataInput, hiddenUnits, initFlag, numEpochs, learningRate)

with open(metricsFileName, 'w') as f:
    f.write(MetricsOutString)


#writing to Output file
np.savetxt(trainOutputFilename, YtrainOutput)
np.savetxt(validOutputFilename, YvalidOutput)

end = time.time()
print("Total time taken:")
print(end - start)

print("here")
a = np.array([[1,1,3],[2,3,3]])
print(np.argmax(a, axis=1))























