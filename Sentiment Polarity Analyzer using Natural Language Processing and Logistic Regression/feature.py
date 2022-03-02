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

    trainInputFilename = sys.argv[1]
    validationInputFilename = sys.argv[2]
    testInputFilename = sys.argv[3]
    dictInputFilename = sys.argv[4]

    formattedTrainOut = sys.argv[5]
    formattedValidOut = sys.argv[6]
    formattedTestOut = sys.argv[7]
    featureFlag = int(sys.argv[8])
    featureDictInputFilename = sys.argv[9]
#
# trainInputFilename = 'train_data.tsv'
# testInputFilename = 'test_data.tsv'
# validationInputFilename = 'valid_data.tsv'
# dictInputFilename = 'dict.txt'
# featureDictInputFilename = 'word2vec.txt'
# featureFlag = 2
# formattedTrainOut = 'model2_formatted_trainC.tsv'
# formattedTestOut = 'model2_formatted_testC.tsv'
# formattedValidOut = 'model2_formatted_validC.tsv'
# refTrainOut = 'model2_formatted_train.tsv'
# refValidOut = 'model2_formatted_valid.tsv'
# refTestOut = 'model2_formatted_test.tsv'

# load the input data into numpy array
start = time.time()

trainDataInput = np.genfromtxt(trainInputFilename,  dtype='str', delimiter="\t")

#------------------------------------ bag of words -----------------------------------------------#

def bagofwords(dictInputFilename, InputFilename, outputFilename):
    vocabTemp = []
    vocabDictionary = {}
    #read the dict of bag of words into key value pairs
    with open(dictInputFilename) as f:
        lines = f.read().splitlines()
        for line in lines:
            vocabTemp = line.split(" ")
            vocabDictionary[vocabTemp[0]] = vocabTemp[1]

    labels = []
    words = []

    #read and parse input file
    with open(InputFilename, 'r') as file:
        # reading each line
        for i,line in enumerate(file):
            labels.append(line.split("\t")[0])
            pat = r"[^! \.?)(:\",]+"
            #pat = r"[][a-zA-Z0-9\-`='_%&#+>$/]+"
            matches =  re.findall(pat, line.split("\t")[1])
            words.append(matches)

    #create op array
    outputArr = np.empty([len(labels), len(vocabDictionary)+1])
    print("printing")
    print(outputArr[:5,:5])
    #assign first column of array as the labels
    outputArr[:,0] = np.array(labels)

    #if word is found in dictionary, make corresponding column of output as 1
    for i,line in enumerate(words):
        for word in line:
            if word in vocabDictionary.keys():
                outputArr[i, int(vocabDictionary[word])+1] = 1

    np.savetxt(outputFilename, outputArr,fmt = '%s', delimiter='\t', newline='\n')

    print("file done !")

    return outputArr

#------------------------------------ wordtovec -----------------------------------------------#
def word2vec(featureDictInputFilename, InputFilename, outputFilename ):

    featureVectorMapping = []

    #checking the reference output
    with open(featureDictInputFilename) as f:
        lines = f.read().splitlines()
        for line in lines:
            featureVectorMapping.append(line.split("\t"))


    featureVectorMapping = np.array(featureVectorMapping)
    labels = []

    trimmedFile = []
    with open(InputFilename, 'r') as file:
        for line in file:
            labels.append(line.split("\t")[0])
            pat = r"[^! \.?)(:\",]+ "
            matches = [x.strip() for x in re.findall(pat,line.split("\t")[1])]
            trimmedFile.append(matches)
    print('check here')


    # create op array
    outputArr = np.empty([len(labels), featureVectorMapping.shape[1]])
    print("output arr shape for "+ str(InputFilename))
    print(outputArr.shape)
    # assign first column of array as the labels

    for i in range(len(trimmedFile)):

        uniqueWords, countsForEachWord = np.unique(np.array(trimmedFile[i]), return_counts=True)
        wordCount = dict(zip(uniqueWords, countsForEachWord))

        # filtering feature vector for the words
        result = featureVectorMapping[np.isin(featureVectorMapping[:, 0], uniqueWords)]
        weights = [wordCount[x] for x in result[:,0]]
        outputArr[i,1:] = np.average(result[:, 1:].astype('float64'), axis=0, weights=weights)

    outputArr[:,0] = np.array(labels)

    np.savetxt(outputFilename, outputArr, fmt='%f', delimiter='\t', newline='\n')
    print("file done !")

    return outputArr


def checking(refOutFile, outputArr):

    vocabTemp = []
    #checking the reference output

    with open(refOutFile) as f:
        lines = f.read().splitlines()
        for line in lines:
            vocabTemp.append(line.split("\t"))

    vocab = np.array(vocabTemp)
    print(np.sum(vocab.astype('float64')))
    print(np.sum(outputArr.astype('float64')))
    vocab = vocab.astype('float64')
    outputArr = outputArr.astype('float64')

    print(vocab.shape)
    print(outputArr.shape)
    print(np.where(vocab!=outputArr))








### main function

if (featureFlag == 1):

    #traininput
    formattedTrain = bagofwords(dictInputFilename, trainInputFilename, formattedTrainOut)
    formattedValid = bagofwords(dictInputFilename, validationInputFilename, formattedValidOut)
    formattedTest = bagofwords(dictInputFilename, testInputFilename, formattedTestOut)

    #checking
    # checking(refTrainOut, formattedTrain)
    # checking(refTestOut, formattedTest)
    # checking(refValidOut, formattedValid)



elif (featureFlag == 2):

    formattedTrain =  word2vec(featureDictInputFilename, trainInputFilename, formattedTrainOut)
    formattedTest = word2vec(featureDictInputFilename, testInputFilename, formattedValidOut)
    formattedValid= word2vec(featureDictInputFilename, validationInputFilename, formattedTestOut)

    # # #checking
    # checking(refTrainOut, formattedTrain)
    # checking(refTestOut, formattedTest)
    # checking(refValidOut, formattedValid)

end = time.time()
print("Total time taken:")
print(end - start)























