import numpy as np
import csv
import math
import random
import sys

# if __name__ == "__main__":
#
#     inputTrainingFileName = sys.argv[1]
#     inputTestFileName = sys.argv[2]
#     max_depth = int(sys.argv[3])
#
#     trainingOutputName = sys.argv[4]
#     testOutputName = sys.argv[5]
#     metricsFileName = sys.argv[6]
#
# print(inputTrainingFileName)
# print(inputTestFileName)
# print(ColumnToSplit)
# print("type  of column to split :", type(ColumnToSplit))
# print(trainingOutputName)
# print(testOutputName)
# print(metricsFileName)

#function to calculate mutual information and decide the best attribute to split on
def mutualInformation(trainDataInput, prevAttributes):

    x_train = trainDataInput[0:, :-1]
    y_train = trainDataInput[0:, -1]

    unique_labels_y, counts_y = np.unique(y_train, return_counts=True)
    n = len(y_train)




    # Mutual Info : I(Y:X) = H(Y) - H(Y|X)
    # H(Y)

    if len(counts_y) == 1:

        entropy_y = 0
    else:
        entropy_y = (-1 / n) * ((counts_y[0]) * math.log2(counts_y[0] / n) + (counts_y[1]) * math.log2(counts_y[1] / n))
    numCol = len(x_train[0])
    mutualInfoDict = {}


    # (Y|X)
    for col in range(numCol):

        if col in prevAttributes:
            entropycolname = 'entropy_y_given_x_' + str(col)
            mutualInfoDict[entropycolname] = None
            continue
        else:

            unique_labels_x, counts_x = np.unique(x_train[0:, col], return_counts=True)
            label_entropy_dict = {}

            for i in unique_labels_x:

                entropylabelvaluename = 'entropy_y_given_x_' + str(col) + '_' + str(i)
                y_given_x = y_train[(x_train[0:, col] == i)]  # subset of y given

                unique_labels_y_given_x, counts_y_given_x = np.unique(y_given_x, return_counts=True)
                x_count = sum(counts_y_given_x)
                unique_labels_y_given_x = list(unique_labels_y_given_x)
                counts_y_given_x = list(counts_y_given_x)

                if len(unique_labels_y_given_x) == 1:
                    label_entropy_dict[entropylabelvaluename] = 0

                else :
                    # H(Y|X1 = 'x') - for a single unique label inside the col

                    entropy_y_given_x_label = (-1 / x_count) * (
                                (counts_y_given_x[0]) * math.log2(counts_y_given_x[0] / x_count) +
                                (counts_y_given_x[1]) * math.log2(counts_y_given_x[1] / x_count))

                    label_entropy_dict[entropylabelvaluename] = entropy_y_given_x_label

            # H(Y|X1) for a single col
            entropy_cols = list(label_entropy_dict.keys())
            entropy_vals = list(label_entropy_dict.values())

            entropycolname = 'entropy_y_given_x_' + str(col)

            if len(unique_labels_x) == 1:
                mutualInfoDict[entropycolname] = 0

            else:

                entropy_y_given_x = (1 / sum(counts_x)) * (entropy_vals[0] * counts_x[0] + entropy_vals[1] * counts_x[1])
                mutualInfoDict[entropycolname] = entropy_y - entropy_y_given_x


    mutualInfoCols = list(mutualInfoDict.keys())
    mutualInfoVals = list(mutualInfoDict.values())

    max_MI = max([num for num in mutualInfoVals if num is not None])

    if max_MI > 0 :
        for key, value in mutualInfoDict.items():
            if value == max_MI :
                splitIndex = int(mutualInfoCols.index(key))
    else :
        splitIndex = None

     #giving column to split on
    return splitIndex

#function to calculate error rate for both train and test predictions
def errorrate( y_train, y_pred_train, y_test, y_pred_test):

    y_train = y_train[0:, -1]
    y_test = y_test[0:, -1]


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


#function for error rate
def error(data):

    label = data[0:, -1]
    unique_labels, counts = np.unique(label, return_counts=True)
    n = len(label)
    error = min(counts) / n
    return error

#function for majority vote
def majority_vote(data):

    label = data[0:, -1]
    unique_labels, counts = np.unique(label, return_counts=True)
    unique_labels = list(unique_labels)
    counts = list(counts)
    majority_vote = max(counts)

    majorityVoteDict = dict(zip(unique_labels, counts))
    for key, value in majorityVoteDict.items():
        if value == majority_vote:
            majority_vote_label = key

    return majority_vote_label

#initialising a Node

class Node:
    def __init__(self, data, maxDepth, currentDepth, prevAttributes):

        self.left = None
        self.right = None
        self.data = data

        if prevAttributes is None:
            self.prevAttributes = []
        else:
            self.prevAttributes = prevAttributes

        if currentDepth is None:
            self.depth = 0
        else:
            self.depth = currentDepth



    def growTree(self, data, maxDepth):

        self.depth = self.depth + 1

        if len(self.prevAttributes) == colNames.size - 1:
            self.nodeAttribute = self.prevAttributes[-1]
        else:
            # get attribute to split on
            self.nodeAttribute = mutualInformation(data, self.prevAttributes)
            if self.nodeAttribute != None:
                self.prevAttributes.append(self.nodeAttribute)


        # get unique labels of that attribute
        unique_labels, countsInEachLabel = np.unique(data[0:, self.nodeAttribute], return_counts=True)


        # initialize values for the node

        # assigning branch labels
        self.leftbranch = unique_labels[0]
        self.rightbranch = unique_labels[1]

        if self.depth < maxDepth and len(self.prevAttributes) < colNames.size - 1 and self.nodeAttribute is not None:
            self.majorityPredictionLeft = None
            self.majorityPredictionRight = None
            #left node
            self.left = Node(data[data[0:, self.nodeAttribute] == self.leftbranch], maxDepth, self.depth, self.prevAttributes)
            self.left.growTree(data[data[0:, self.nodeAttribute] == self.leftbranch], maxDepth)


            #right node
            self.right = Node(data[data[0:, self.nodeAttribute] == self.rightbranch], maxDepth, self.depth, self.prevAttributes)
            self.right.growTree(data[data[0:, self.nodeAttribute] == self.rightbranch], maxDepth)

        else:

            if self.nodeAttribute is not None:

                self.majorityPredictionLeft = majority_vote(data[data[0:, self.nodeAttribute] == self.leftbranch])
                self.majorityPredictionRight = majority_vote(data[data[0:, self.nodeAttribute] == self.rightbranch])

            else:
                self.majorityPredictionRight = majority_vote(data)
                self.majorityPredictionLeft = majority_vote(data)




        return self



#defining the train function
def train(trainDataInput, maxDepth):

    #initialize root node
    root = Node(trainDataInput, maxDepth, None, None)

    #call grow_tree to learn the decision tree
    tree = root.growTree(trainDataInput, maxDepth)
    return tree

def predict(data, node):

    if node.nodeAttribute is None:
        attribute = node.prevAttributes[-1]
    else:
        attribute = node.nodeAttribute

    if node is None:
        print("nothing")

    elif node.left is None:

        if data[attribute] == node.leftbranch:
            return node.majorityPredictionLeft
        else:
            return node.majorityPredictionRight

    elif node.right is None:

        if data[attribute] == node.rightbranch:
            return node.majorityPredictionRight
        else:
            return node.majorityPredictionLeft

    else:

        if data[attribute] == node.leftbranch:
            node = node.left
        elif data[attribute] == node.rightbranch:
            node = node.right
        return predict(data, node)

def PrintTree(node, branch):


        if node.nodeAttribute is not None:
            uniqueOutputLabels, countsInEachOutputLabel = np.unique(node.data[node.data[0:, node.nodeAttribute] == branch][0:, -1], return_counts=True)
            attribute = node.nodeAttribute
        else:
            uniqueOutputLabels, countsInEachOutputLabel = np.unique(node.data[:,-1], return_counts= True)
            attribute = node.prevAttributes[-1]

        if uniqueOutputLabels.size == 1:
            missingvar = list(set(predictionLabels) - set(uniqueOutputLabels))
            uniqueOutputLabels = np.append(uniqueOutputLabels, missingvar, axis=0)
            countsInEachOutputLabel = np.append(countsInEachOutputLabel, 0)

        print('| ' * node.depth, columnNameDict[attribute], ' = ', branch, ': [',
              countsInEachOutputLabel[0],
              ' ', uniqueOutputLabels[0], '/', countsInEachOutputLabel[1], ' ', uniqueOutputLabels[1], ']')


def PrintRecurse(node):

        if node.majorityPredictionLeft is None:
            PrintTree(node, node.leftbranch)
            PrintRecurse(node.left)
        if node.majorityPredictionRight is None:
            PrintTree(node, node.rightbranch)
            PrintRecurse(node.right)
        else:
            PrintTree(node, node.leftbranch)
            PrintTree(node, node.rightbranch)























def test(data, node):

    predictions = np.array([])

    for row in data:

        result = predict(row, node)
        predictions = np.append(predictions, result)


    return predictions



#main function

inputTrainingFileName = 'politicians_train.tsv'
inputTestFileName = 'politicians_test.tsv'
testOutputName = 'politicians_4_test.LABELS'
trainingOutputName = 'politicians_4_train.LABELS'
metricsFileName = 'politicians_4_metrics.txt'
max_depth = 4


trainDataInput = np.genfromtxt(inputTrainingFileName, skip_header=1, dtype='str', delimiter="\t")
testDataInput = np.genfromtxt(inputTestFileName, skip_header=1, dtype='str', delimiter="\t")

#get column attributes in a dict
colNames = np.genfromtxt(inputTrainingFileName, max_rows= 1, dtype='str', delimiter="\t")


columnNameDict = dict(list(enumerate(colNames)))
predictionLabels = list(np.unique(trainDataInput[0:, -1]))



#call training function
tree = train(trainDataInput, max_depth)
print("Training done")
print(tree.left.right.prevAttributes)
tree
PrintRecurse(tree)

#predict function
predictionsTestData = test(testDataInput, tree)
predictionsTrainData = test(trainDataInput, tree)



#error calculation
TestErrorRate, TrainErrorRate = errorrate( trainDataInput, predictionsTrainData,testDataInput, predictionsTestData)


#writing to Output file
np.savetxt(testOutputName, predictionsTestData, fmt="%s")
np.savetxt(trainingOutputName, predictionsTrainData, fmt="%s")

line1 = 'error(train): ' + str(TrainErrorRate)
line2 = 'error(test): ' + str(TestErrorRate)

with open(metricsFileName, 'w') as f:
    f.write(line1)
    f.write('\n')
    f.write(line2)