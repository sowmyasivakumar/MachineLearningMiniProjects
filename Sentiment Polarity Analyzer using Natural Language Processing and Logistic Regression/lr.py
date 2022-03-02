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


if __name__ == "__main__":

    trainInput = sys.argv[1]
    validInput = sys.argv[2]
    testInput = sys.argv[3]
    dictInputFilename = sys.argv[4]

    trainingOutputName = sys.argv[5]
    testOutputName = sys.argv[6]
    metricsFileName = sys.argv[7]
    num_epochs = int(sys.argv[8])


#
# trainInput = 'model2_formatted_train.tsv'
# testInput = 'model2_formatted_test.tsv'
# validInput = 'model2_formatted_valid.tsv'
#
# refTrainOut = 'model2_train_out.LABELS'
# refTestOut = 'model2_test_out.LABELS'
#
# trainingOutputName = 'model2_train_out.LABELS'
# testOutputName = 'model2_test_out.LABELS'
# metricsFileName = 'model2_metrics_out.txt'
#
#
# num_epochs = 500

# load the input data into numpy array
start = time.time()


def sigmoid(X, theta):
    Z = 1 / (1 + np.exp(-np.dot(theta.T, X)))
    return Z

def loglikelihood(X,Y,theta):
    s = -1*Y*np.dot(theta.T, X)
    t = np.log(1 + np.exp(np.dot(theta.T, X)))
    val = s+t
    return val


#----------------------------------------------------------------------------------------------------------------
def train(trainDataInput, num_epochs):

    # store input labels
    Y = trainDataInput[:, 0]
    Y = Y.reshape(Y.shape[0], 1)

    # take input from 1st col and fold 1 to the 0th col
    X = np.column_stack((np.ones_like(Y), trainDataInput[:, 1:]))

    # initialise all params as zero including bias
    theta = np.zeros((X.shape[1], 1))
    alpha = 0.01
    N = len(X)
    print('Theta shape', theta.shape)
    print('Y shape', Y.shape)
    #
    negativeLogLikelihood = []
    for k in range(0,num_epochs):
        print(k)
        sum = 0
        for i in range(0,len(X)):

                Z = np.array(alpha/N)*X[i]*(Y[i] - 1/(1 + np.exp(-np.dot(theta.T, X[i]))))
                Z = Z.reshape(Z.shape[0], 1)
                theta = theta + Z
                res = loglikelihood(X[i],Y[i],theta)
                sum = sum + res

        negativeLogLikelihood.append(sum/len(X))

    obj = np.array(negativeLogLikelihood)
    np.savetxt('NegativeLogLikelihood.csv', obj, delimiter= ",")

    print("training done !")
    return theta
#----------------------------------------------------------------------------------------------------------------

def test(DataInput, theta):


    Y = DataInput[:, 0]
    Y = Y.reshape(Y.shape[0], 1)


    # take input from 1st col and fold 1 to the 0th col
    X = np.column_stack((np.ones_like(Y), DataInput[:, 1:]))


    sigmoidOutput = np.apply_along_axis(sigmoid, 1, X, theta)

    Ypred = np.where(sigmoidOutput > 0.5, 1, 0)

    print("testing done!")
    return Ypred



#function to calculate error rate for both train and test predictions
def errorrate( traindata, YpredTrain, testdata, YpredTest):

    print("error function")
    Ytrain = traindata[:, 0]
    Ytest = testdata[:, 0]
    Ytrain = Ytrain.astype('int32')
    Ytest = Ytest.astype('int32')

    Ytrain = Ytrain.reshape(Ytrain.shape[0], 1)
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    print(Ytrain.shape)
    print(YpredTrain.shape)



    TotalNoOfValues = Ytest.size
    TestPredErrorCount = 0
    TrainPredErrorCount = 0

    #test error rate
    for x, y in np.nditer([Ytest, YpredTest]):
        if x != y:
            TestPredErrorCount += 1

    TestPredErrorRate = TestPredErrorCount/TotalNoOfValues
    print("Test Error Rate :", TestPredErrorRate)

    # train error rate

    TotalNoOfValues = Ytrain.size
    for x, y in np.nditer([Ytrain, YpredTrain]):
        if x != y:
            TrainPredErrorCount += 1

    TrainPredErrorRate = TrainPredErrorCount / TotalNoOfValues
    print("Train Error Rate :", TrainPredErrorRate)

    return TestPredErrorRate, TrainPredErrorRate




def checking(testfile, refFile):
    print("checking " + str(refFile) )
    checking = []
    # checking the reference output

    print(np.unique(testfile, return_counts=True))
    with open(refFile) as f:
        lines = f.read().splitlines()
        for line in lines:
            checking.append(line)

    vocab = np.array(checking)
    vocab = vocab.astype('int64').reshape(vocab.shape[0], 1)
    print(vocab.shape)

    testfile = testfile.astype('int64')
    print(np.unique(vocab, return_counts=True))
    print("Check for match here:")
    print(np.where(vocab != testfile))

# -------------------------reading text files -----------------------------


trainDataInput = np.genfromtxt(trainInput, delimiter="\t")
testDataInput = np.genfromtxt(testInput, delimiter="\t")
validDataInput = np.genfromtxt(validInput, delimiter="\t")

print('Train input shape' , trainDataInput.shape)
#--------------------------prediction function ---------------------------------


theta = train(validDataInput, num_epochs)

#testing
YpredTrain = test(trainDataInput, theta)
YpredTest = test(testDataInput, theta)
Ypredvalid = test(validDataInput, theta)

traintime = time.time()
print("Total training time taken:")
print(traintime - start)

#check file
# checking(YpredTrain, refTrainOut)
# checking(YpredTest, refTestOut)

end = time.time()
print("Total time taken:")
print(end - start)

Ytrain = trainDataInput[:, 0]
Ytest = testDataInput[:, 0]

#error calculation
TestErrorRate, TrainErrorRate = errorrate( trainDataInput, YpredTrain,testDataInput, YpredTest)



#writing to Output file
np.savetxt(testOutputName, YpredTest, fmt="%s")
np.savetxt(trainingOutputName, YpredTrain, fmt="%s")

line1 = 'error(train): ' + str(TrainErrorRate)
line2 = 'error(test): ' + str(TestErrorRate)

with open(metricsFileName, 'w') as f:
    f.write(line1)
    f.write('\n')
    f.write(line2)