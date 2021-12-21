import numpy as np
import csv
import math
import random
import sys

if __name__ == "__main__":

    inputFileName = sys.argv[1]
    outputFileName = sys.argv[2]

#
# print(inputTrainingFileName)
# print(inputTestFileName)
# print(ColumnToSplit)
# print("type  of column to split :", type(ColumnToSplit))
# print(trainingOutputName)
# print(testOutputName)
# print(metricsFileName)
#
# inputFileName = 'small_train.tsv'
# outputFileName = 'small_inspect.txt'
#format the input into numpy array
trainDataInput = np.genfromtxt(inputFileName, skip_header=1,  dtype='str', delimiter="\t")

#get required columns and info for calculation
label = trainDataInput[0:, -1]

unique_labels, counts = np.unique(label, return_counts=True)
n = len(label)


#entropy calc
if len(counts) == 1:
    entropy = 0
else :
    entropy = (-1/n)*((counts[0])*math.log2(counts[0]/n) + (counts[1])*math.log2(counts[1]/n))

line1 = "entropy: " + str(entropy)


#error_calc
error = min(counts)/n
line2 = "error: " + str(error)

#write into output file
with open(outputFileName, 'w') as f:
    f.write(line1)
    f.write('\n')
    f.write(line2)

print("completed!")









