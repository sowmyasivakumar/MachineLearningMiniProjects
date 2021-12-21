import numpy as np
import csv
import math
import random
import sys
import re
import filecmp
import time

#
#
if __name__ == "__main__":

    trainInputFilename = sys.argv[1]
    wordIndexData = sys.argv[2]
    tagIndexData = sys.argv[3]

    outputHmmInit = sys.argv[4]
    outputHmmEmit = sys.argv[5]
    outputHmmTrans = sys.argv[6]

#
#initializing manually

# trainInputFilename = 'toy_data/train.txt'
# validationInputFilename = 'toy_data/validation.txt'
# tagIndexData = 'toy_data/index_to_tag.txt'
# wordIndexData = 'toy_data/index_to_word.txt'
#
# outputHmmInit = 'toy_data/hmminit.txt'
# outputHmmEmit = 'toy_data/hmmemit.txt'
# outputHmmTrans = 'toy_data/hmmtrans.txt'


# checkOutputHmmInitFile = 'toy_output/hmminit.txt'
# checkOutputHmmEmitFile = 'toy_output/hmmemit.txt'
# checkOutputHmmTransFile = 'toy_output/hmmtrans.txt'

def checkallfiles(hmmtrans, hmmemit, hmminit, checkOutputHmmTransFile, checkOutputHmmEmitFile, checkOutputHmmInitFile):

   checkOutputHmmInit = np.genfromtxt(checkOutputHmmInitFile)
   checkOutputHmmEmit = np.genfromtxt(checkOutputHmmEmitFile)
   checkOutputHmmTrans = np.genfromtxt(checkOutputHmmTransFile)

   checkOutputHmmInit = checkOutputHmmInit.astype('float64')
   checkOutputHmmTrans = checkOutputHmmTrans.astype('float64')
   checkOutputHmmEmit = checkOutputHmmEmit.astype('float64')

   hmmemit1 = hmmemit.astype('float64')
   hmminit1 = hmminit.astype('float64')
   hmmtrans1 = hmmtrans.astype('float64')
   print(hmminit1)
   print(checkOutputHmmInit)

   print('checking init')
   print(np.unique(hmminit1.T == checkOutputHmmInit))
   print('checking emit')
   print(np.unique(hmmemit1 == checkOutputHmmEmit))
   print('checking trans')
   print(np.unique(hmmtrans1 == checkOutputHmmTrans))




# load the input data into numpy array
start = time.time()

trainDataInput = np.genfromtxt(trainInputFilename, delimiter='\t', dtype= 'str')
tagIndex = np.genfromtxt(tagIndexData, dtype= 'str')
wordIndex = np.genfromtxt(wordIndexData, dtype= 'str')



with open(trainInputFilename) as f:
   contentsT = f.read()
   arrT = contentsT.replace('\n\n', '\nbreak\tbreak\n')
   s = arrT.split('break\tbreak')
   firstWordWithTag = [k.strip().split('\n')[0] for k in s]
   firstTags = [g.split('\t')[1] for g in firstWordWithTag]
   firstTags = np.array([firstTags])



#initialize hmminit (initial probability)
hmminit = np.ones((len(tagIndex), 1))
uniqueTags, counts = np.unique(firstTags, return_counts=True)
uniqueTagCount = dict(zip(uniqueTags, counts))

for index, i  in enumerate(tagIndex):
   if (i in uniqueTagCount.keys()):
      hmminit[index] = uniqueTagCount[i]+1


hmminit = hmminit/ sum(hmminit)

print('hmminit done')
#initialize hmmemit(emission probabilities of p(x|y)

hmmemit = np.ones((len(tagIndex), len(wordIndex)))


# for x, y in np.nditer([tagIndex, wordIndex]):
#    occurrences = np.count_nonzero((trainDataInput[:, 1] == x) & (trainDataInput[:, 0] == y))
#    hmmemit[np.argwhere(x)][np.argwhere(y)] = occurrences+1

concatarray = np.char.add(np.char.add(trainDataInput[:, 0],'\t'), trainDataInput[:, 1])
xgiveny, countsxgiveny = np.unique(concatarray, return_counts=True)
uniqueComboCount = dict(zip(xgiveny, countsxgiveny))



for i, tag in enumerate(tagIndex):
   for j,word in enumerate(wordIndex):
      g = word + '\t' + tag
      if (g in uniqueComboCount):
         countval = uniqueComboCount[g]
         hmmemit[i][j] = countval+1

# for i, tag in enumerate(tagIndex):
#    print(i)
#    for j,word in enumerate(wordIndex):
#
#       occurrences = np.count_nonzero((trainDataInput[:, 1] == tag) & (trainDataInput[:, 0] == word))
#       hmmemit[i][j] = occurrences+1


hmmemit = hmmemit/hmmemit.sum(axis=1)[:,None]


print('hmmemit done')

#initialize hmmtrans - transition probabilities(p(y|y-1 ))
hmmtrans = np.empty([len(tagIndex), len(tagIndex)])

with open(trainInputFilename) as f:
   contents = f.read()
   arr = contents.replace('\n\n', '\nbreak\tbreak\n')

s = arr.split('\n')


tags = np.loadtxt(s,delimiter='\t', dtype='str')

tagleads = np.roll(tags[:,1], -1)
tagArray = np.column_stack((tags[:,1],tagleads))
tagArray = tagArray[:-1]

for i, tagOld in enumerate(tagIndex):
   for j, tagNew in enumerate(tagIndex):
       hmmtrans[j,i] = len(tagArray[(tagArray[:,0] == tagOld) & (tagArray[:,1] == tagNew)])+1

print('hmmtrans done')
hmmtrans = hmmtrans.T/hmmtrans.T.sum(axis=1)[:,None]

# checkallfiles(hmmtrans, hmmemit, hmminit, checkOutputHmmTransFile, checkOutputHmmEmitFile, checkOutputHmmInitFile)

np.savetxt(outputHmmInit, hmminit, fmt='%.18e')
np.savetxt(outputHmmEmit, hmmemit, fmt='%.18e')
np.savetxt(outputHmmTrans, hmmtrans, fmt='%.18e')
































