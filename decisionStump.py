import numpy as np
import csv
import math
import random
import sys

if __name__ == "__main__":

    inputTrainingFileName = sys.argv[1]
    inputTestFileName = sys.argv[2]
    ColumnToSplit = int(sys.argv[3])

    trainingOutputName = sys.argv[4]
    testOutputName = sys.argv[5]
    metricsFileName = sys.argv[6]

print(inputTrainingFileName)
print(inputTestFileName)
print(ColumnToSplit)
print("type  of column to split :", type(ColumnToSplit))
print(trainingOutputName)
print(testOutputName)
print(metricsFileName)

#function to train data
def train(X_train, Y_train, trainLabelA, trainLabelB, predictionLabelA, predictionLabelB):

    trainA_predA = 0
    trainA_predB = 0
    trainB_predA = 0
    trainB_predB = 0

    for x, y in np.nditer([x_train, y_train]):
        if x == trainLabelA and y == predictionLabelA:
            trainA_predA += 1
        elif x == trainLabelA and y == predictionLabelB:
            trainA_predB += 1
        elif x == trainLabelB and y == predictionLabelA:
            trainB_predA += 1
        elif x == trainLabelB and y == predictionLabelB:
            trainB_predB += 1



    # Determining Majority for Node1
    if trainA_predA > trainA_predB:
        majorityvoteA = predictionLabelA
    elif trainA_predA < trainA_predB:
        majorityvoteA = predictionLabelB
    else:
        majorityvoteA = random.choice([predictionLabelA, predictionLabelB])

    # print(trainA_predA, trainA_predB)
    # print("Majority Vote A ::",  majorityvoteA)

    # Determining Majority for Node2
    if trainB_predA > trainB_predB:
        majorityvoteB = predictionLabelA
    elif trainB_predA < trainB_predB:
        majorityvoteB = predictionLabelB
    else:
        majorityvoteB = random.choice([predictionLabelA, predictionLabelB])

    # print(trainB_predB, trainB_predA)
    # print("Majority Vote B ::",  majorityvoteB)

    return majorityvoteA, majorityvoteB


# function to test data
def test(x_test, y_test, testLabelA, testLabelB, majorityvoteA, majorityvoteB):

    y_pred = np.array([])

    for x in x_test:

        if x == testLabelA:
            y_pred = np.append(y_pred, majorityvoteA)
        elif x == testLabelB:
            y_pred = np.append(y_pred, majorityvoteB)

    return y_pred


#function to calculate error rate for both train and test predictions
def errorrate(y_test, y_train, y_pred_train, y_pred_test):

    TotalNoOfValues = y_test.size
    TestPredErrorCount = 0
    TrainPredErrorCount = 0

    #test error rate
    for x, y in np.nditer([y_test, y_pred_test]):
        if x != y:
            TestPredErrorCount += 1

    TestPredErrorRate = TestPredErrorCount/TotalNoOfValues
    print("Test Error Rate :", TestPredErrorRate)

    # train error rate

    TotalNoOfValues = y_train.size
    for x, y in np.nditer([y_train, y_pred_train]):
        if x != y:
            TrainPredErrorCount += 1

    TrainPredErrorRate = TrainPredErrorCount / TotalNoOfValues
    print("Train Error Rate :", TrainPredErrorRate)

    return TestPredErrorRate, TrainPredErrorRate


# trainDataInputTsv = open("politicians_train.tsv")
# inputTrainingFileName = "politicians_train.tsv"
trainDataInput = np.genfromtxt(inputTrainingFileName, skip_header=1, usecols=(ColumnToSplit, -1), dtype='str', delimiter="\t")

# testDataInputTsv = open("politicians_test.tsv")
# inputTestFileName = "politicians_test.tsv"
testDataInput = np.genfromtxt(inputTestFileName, skip_header=1, usecols=(ColumnToSplit, -1), dtype='str', delimiter="\t")

# trainLabelA implies anti_satellite_test_ban = y and trainlabelB assumes anti_satellite_test_ban = n
if inputTrainingFileName == 'inputs/small_train.tsv' or inputTestFileName == 'inputs/small_test.tsv':
    print("I'm here")
    trainLabelA = 'y'
    trainLabelB = 'n'
    testLabelA = 'y'
    testLabelB = 'n'
    predictionLabelA = 'republican'
    predictionLabelB = 'democrat'

elif inputTrainingFileName == 'inputs/education_train.tsv' or inputTestFileName == 'inputs/education_test.tsv':
    print("I'm here in education")
    trainLabelA = 'notA'
    trainLabelB = 'A'
    testLabelA = 'notA'
    testLabelB = 'A'
    predictionLabelA = 'notA'
    predictionLabelB = 'A'

elif inputTrainingFileName == 'inputs/politicians_train.tsv' or inputTestFileName == 'inputs/politicians_test.tsv':
    print("I'm here in politician")
    trainLabelA = 'y'
    trainLabelB = 'n'
    testLabelA = 'y'
    testLabelB = 'n'
    predictionLabelA = 'republican'
    predictionLabelB = 'democrat'


splitIndex = 0
x_train = trainDataInput[0:, splitIndex]
y_train = trainDataInput[0:, -1]


x_test = testDataInput[0:, splitIndex]
y_test = testDataInput[0:, -1]

#training
majorityvoteA, majorityvoteB = train(x_train, y_train, trainLabelA, trainLabelB, predictionLabelA, predictionLabelB)


#test data predictions
y_pred_test = test(x_test, y_test, testLabelA, testLabelB, majorityvoteA, majorityvoteB)
# print("Returned data test predictions")

#train data predictions
y_pred_train = test(x_train, y_train, testLabelA, testLabelB, majorityvoteA, majorityvoteB)

#error rate calculation
TestErrorRate, TrainErrorRate = errorrate(y_test, y_train, y_pred_train, y_pred_test)

#writing output in required format
outputfilename = inputTestFileName[:-4].split("_")
print(outputfilename)

# metrics file
# metricsFileName = outputfilename[0]+'_'+str(splitIndex)+'_metrics.txt'
print(metricsFileName)
line1 = 'error(train): ' + str(TrainErrorRate)
line2 = 'error(test): ' + str(TestErrorRate)

with open(metricsFileName, 'w') as f:
    f.write(line1)
    f.write('\n')
    f.write(line2)

# training file
# trainingOutputName = outputfilename[0]+'_'+str(splitIndex)+'_train.LABELS'
np.savetxt(trainingOutputName, y_pred_train, fmt="%s")

#test file output
# testOutputName = outputfilename[0]+'_'+str(splitIndex)+'_test.LABELS'

np.savetxt(testOutputName, y_pred_test, fmt="%s")