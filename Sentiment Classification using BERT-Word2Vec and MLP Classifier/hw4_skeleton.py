# Importing the required packages

import os
import re

import pandas as pd
import numpy as np

import gensim
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


from itertools import compress
import collections

# first step is to preprocess the data

# Importing the datasets
train = pd.read_csv('train.csv', error_bad_lines=False)
valid = pd.read_csv('valid.csv', error_bad_lines=False)
test = pd.read_csv('test.csv', error_bad_lines=False)

# select "utterance" and "context" as your X and y
# only select {'sad', 'jealous', 'joyful', 'terrified'} categories
# use pandas.loc function to do it!


X_train =


X_test =

# Getting the train labels; this will be used for SGD classifier
train_labels_unique = list(train['context'].unique())
label_mapper = {}
num = 0
for label in train_labels_unique:
    label_mapper[label] = num
    num += 1


train_labels = list(train['context'])
train_labels_encoded = []
for label in train_labels:
    train_labels_encoded.append(label_mapper[label])
    
# Getting test labels
labels_test = list(test['context'])
labels_encoded_test = []
for label in labels_test:
    labels_encoded_test.append(label_mapper[label])
labels_encoded_test = np.array(labels_encoded_test)

# note train and test labels are in train_labels_encoded and labels_encoded_test

# data preprocessing, remove punctuations from the sentence
# DO NOT REMOVE STOPWORDS here
# follow this article should be good to go 
# https://medium.com/@arunm8489/getting-started-with-natural-language-processing-6e593e349675
# https://machinelearningmastery.com/clean-text-machine-learning-python/


train_data_list_cleaned = 
test_data_list_cleaned = 

### 2
## Converting the utterances into a sparse bag-of-words 

# use sklearn should be good
train_count_vectorizer = CountVectorizer()
X = train_count_vectorizer.fit_transform(train_data_list_cleaned)
encoding = X.toarray()

# Converting counts to binary resukt
for arr in encoding:
    arr[arr > 0] = ?
    
# now you have the sparse encoding


### 3 
## The shortcomings with the previous representation are ?

# so we should possibly remove stop words.
# Getting the list of stopwords and appending additional words to it
stopwords_list = list(set(stopwords.words('english')))
stopwords_list.extend(['comma', ''])  

# remove the tokens in the stopwords list from utterance

train_data_stop_removed = 
test_data_stop_removed = 

# Creating the bag of words encoding again  
train_count_vectorizer = CountVectorizer()
X_train = train_count_vectorizer.fit_transform(train_data_stop_removed)

train_one_hot_encoding = X_train.toarray()

for arr in train_one_hot_encoding:
    arr[arr > 0] = ?



### 4. Normalization
# Normalizing the training data using tfidf transformer 
# let us use sklearn again

train_tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
train_embedding_tfidf_transformer = train_tfidf_transformer.?


### 5. Building an SGD Classifier
X_train = train_embedding_tfidf_transformer
y_train = np.array(train_labels_encoded)

clf = this place should be a SGD classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
clf.fit(X_train, y_train)

# you can evaluate the training accuracy by predicted result and true label y_train


# Using training data vocabulary on test data so that the features are consistent    
test_count_vectorizer = CountVectorizer(vocabulary = train_count_vectorizer.get_feature_names())
X_test = test_count_vectorizer.fit_transform(test_data_stop_removed)

test_one_hot_encoding = X_test.toarray()

for arr in test_one_hot_encoding:
    arr[arr > 0] = 1

# Normalizing the test data  
test_tfidf_transformer = TfidfTransformer(smooth_idf=False,use_idf=True)
test_embedding_tfidf_transformer = test_tfidf_transformer.fit_transform(test_one_hot_encoding)

# Getting predictions on test data
test_predicted_labels = clf.predict(test_embedding_tfidf_transformer)


# do some evaluation on the test set
print('Test accuracy :', np.mean(labels_encoded_test == test_predicted_labels))

f1_score_vector = f1_score(labels_encoded_test, test_predicted_labels, average=None)

print('F1 score :', np.mean(labels_encoded_test == test_predicted_labels))

print('Confusion matrix :', confusion_matrix(labels_encoded_test, test_predicted_labels))

print('f1 score using SGD classifier is :', np.mean(f1_score_vector))


### 6. Classifier using pretrained embeddings

# Tokenizing the data
train_tokens = [nltk.word_tokenize(sentences) for sentences in train_data_stop_removed]
train_y = np.array(train_labels_encoded)

test_tokens = [nltk.word_tokenize(sentences) for sentences in test_data_stop_removed]
test_y = np.array(labels_encoded_test)

# Loading the pretrained word2vec model from Google
# download the model here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# follow this article to do it
# https://towardsdatascience.com/using-word2vec-to-analyze-news-headlines-and-predict-article-success-cdeda5f14751


### 7. Classifier using pretrained BERT

from transformers import DistilBertTokenizer, DistilBertModel
# load the tokenizer and the model of distilbert-base-uncased
tokenizer = 
model =

# tokenize the text, then input the tokens (and masks) into the model to get the output

# use the BERT output to train a MLP classifier
