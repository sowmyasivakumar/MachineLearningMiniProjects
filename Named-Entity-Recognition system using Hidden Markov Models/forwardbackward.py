import numpy as np
import csv
import math
import random
import sys
import re
import filecmp
import time
#
# -------------------------------- inputs to be taken ------------------------------------
#  <train input>
# <validation input> <test input> <dict input> <formatted train out>
# <formatted validation out> <formatted test out> <feature flag>
# <feature dictionary input>.
#
# if __name__ == "__main__":
#
#     validationInputFilename = sys.argv[1]
#     wordIndexData = sys.argv[2]
#     tagIndexData = sys.argv[3]
#
#     outputHmmInit = sys.argv[4]
#     outputHmmEmit = sys.argv[5]
#     outputHmmTrans = sys.argv[6]
#     predictedOutputFile = sys.argv[7]
#     metricsFileName = sys.argv[8]


# # initializing manually
#
trainInputFilename = 'en_data/train.txt'
validationInputFilename = 'en_data/validation.txt'
tagIndexData = 'en_data/index_to_tag.txt'
wordIndexData = 'en_data/index_to_word.txt'

outputHmmInit = 'en_output/hmminit.txt'
outputHmmEmit = 'en_output/hmmemit.txt'
outputHmmTrans = 'en_output/hmmtrans.txt'
metricsFileName = 'en_data/metrics.txt'
predictedOutputFile = 'en_data/predicted.txt'
#

def betaCalc(valWords,i, indices,beta, tagIndex, pi, A, B):
    if (i == len(valWords)-1):
        beta[i] = np.log(1)
        return beta, i -1
    else :
        nextBeta, i = betaCalc(valWords, i + 1, indices, beta, tagIndex, pi, A, B)
        for j in (range(len(beta[i]))):
            vi = []
            for k in (range(0, len(tagIndex))):
                vi.append(nextBeta[i + 1][k] + np.log(B[k][j]) + np.log(A[k,int(indices[i+1])]))

            t = logsumexp(vi)

            beta[i][j] = t



        return beta, i - 1

#backward algo
def backward(validationDataInput, tagIndex, wordIndex, pi, A, B):
    # initialize beta matrix
    beta = np.empty([len(validationDataInput), len(tagIndex)])

    valWords = validationDataInput[:, 0]
    # indices = np.where(wordIndex[:, None] == valWords[None, :])[1]
    indices = np.array([])
    for word in valWords:
            indices = np.append(indices,int(np.where(wordIndex==word)[0]))

    sequence = np.column_stack((valWords, indices))

    i = 0
    beta,u = betaCalc(valWords,i, indices,beta, tagIndex, pi, A, B)
    return beta



def logsumexp(vi):
    m = max(vi)
    new_vi = vi-m
    return m + np.log(np.sum(np.exp(new_vi)))

def alphaCalc(valWords,i, indices,alpha, tagIndex, pi, A, B):

    if (i==0):


       alpha[i] = np.log(pi) + np.log(A[:,int(indices[i])])

       return alpha, i+1

    else :

        prevAlpha, i = alphaCalc(valWords, i-1, indices, alpha, tagIndex, pi, A, B)

        for j in range(len(alpha[i])):


            vi = []
            for k in range(0, len(tagIndex)):
                vi.append(prevAlpha[i - 1][k] + np.log(B[k][j]))

            t = logsumexp(vi)
            alpha[i][j] = np.log(A[j][int(indices[i])]) + t


        return alpha, i+1


def forward(validationDataInput, tagIndex, wordIndex, pi, A, B):

    # initialize alpha matrix
    alpha = np.empty([len(validationDataInput), len(tagIndex)])

    valWords = validationDataInput[:,0]
    # indices = np.where(wordIndex[:, None] == valWords[None, :])[1]
    indices = np.array([])
    for word in valWords:
            indices = np.append(indices,int(np.where(wordIndex==word)[0]))


    i = len(valWords) - 1
    alpha,u = alphaCalc(valWords,i, indices,alpha, tagIndex, pi, A, B)

    return alpha

def predict(alpha,beta, tagIndex, validationDataInput):

    validationSequence = validationDataInput[:,1]
    predictor = alpha+beta
    predictor = np.exp(alpha)*np.exp(beta)
    print(predictor)

    predMax = np.argmax(predictor, axis = 1)
    predSequence = tagIndex[predMax]

    error = 0
    # test error rate
    for x, y in np.nditer([validationSequence, predSequence]):
        if x != y:
            error += 1

    predSequence = np.column_stack((validationDataInput[:, 0], tagIndex[predMax]))
    accuracy = 1 - error / len(validationSequence)

    return predSequence, accuracy

def prediction(validationDataInput, tagIndex, wordIndex, hmminit, hmmemit, hmmtrans, predOutputString, listLogLikelihoods, listAccuracy):

    #forward algo
    alpha = forward(validationDataInput, tagIndex, wordIndex, hmminit, hmmemit, hmmtrans)
    print('forward algo done')

    #backward algo
    beta = backward(validationDataInput, tagIndex, wordIndex, hmminit, hmmemit, hmmtrans)
    print('backward algo done')

    #predict the sequence
    predSequence, accuracy = predict(alpha, beta, tagIndex, validationDataInput)
    print('predictions:')
    print(predSequence)


    for row in range(0,len(predSequence)):
         predOutputString = predOutputString + str(f'{predSequence[row,0]}\t{predSequence[row,1]}\n')
    predOutputString = predOutputString + '\n'
    listAccuracy = np.append(listAccuracy, accuracy)

    #negative average log likelihood
    logLikelihood = np.log(np.sum(np.exp(alpha[-1])))
    listLogLikelihoods = np.append(listLogLikelihoods, logLikelihood)


    return predOutputString, listLogLikelihoods, listAccuracy

validationDataInput = np.genfromtxt(validationInputFilename, delimiter='\t', dtype= 'str')
tagIndex = np.genfromtxt(tagIndexData, dtype= 'str')
wordIndex = np.genfromtxt(wordIndexData, dtype= 'str')

hmminit = np.genfromtxt(outputHmmInit)
hmmemit = np.genfromtxt(outputHmmEmit)
hmmtrans = np.genfromtxt(outputHmmTrans)
predOutputString = ''
totalLogLikelihoods = np.array([])
totalAccuracy = np.array([])


with open(validationInputFilename) as f:
   contentsT = f.read()
   arrT = contentsT.replace('\n\n', '\nbreak\tbreak\n')
   s = arrT.split('break\tbreak')
   for y,sequence in enumerate(s):
       print(y)
       seq = sequence.strip().split('\n')
       ls = np.array([g.split('\t') for g in seq])
       predOutputString, totalLogLikelihoods, totalAccuracy = prediction(ls, tagIndex, wordIndex, hmminit, hmmemit, hmmtrans, predOutputString, totalLogLikelihoods, totalAccuracy)

   avgAccuracy = np.average(totalAccuracy)
   avgLogLikelihood = np.average(totalLogLikelihoods)




#write metrics file
MetricsOutString = str(f'Average Log-Likelihood: {avgLogLikelihood}\n')
MetricsOutString = MetricsOutString + str(f'Accuracy: {avgAccuracy}\n')
#
with open(metricsFileName, 'w') as f:
    f.write(MetricsOutString)
#
# #write predicted sequence file
# predSequenceString = ''
#
# for row in range(0,len(predSequence)):
#     predSequenceString = predSequenceString + str(f'{predSequence[row,0]}\t{predSequence[row,1]}\n')
#
with open(predictedOutputFile, 'w') as f:
    f.write(predOutputString)