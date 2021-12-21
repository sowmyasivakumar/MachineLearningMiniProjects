# MachineLearningMiniProjects

This repo contains all the mini-projects that were done as part of the 
Intro to Machine Learning course at CMU. All the models have been built entirely ]
from the scratch without the use of libraries like scikitlearn, keras, PyTorch, etc. 
Only the basic libraries such as NumPy, Math, Random have been used. 
There is no usage of Pandas as well. 

#### Decision Trees 

Problem statement : 
To develop a robust decision tree classifier for binary classification based on numerous binary-valued features(categorical variables).

This has been tested on 3 datasets (on 3 different prediction problems):

1. The first task is to predict whether a US politician is a member of the Democrat or Republican
party, based on their past voting history. Attributes (aka. features) are short descriptions of bills
that were voted on, such as Aid to nicaraguan contras or Duty free exports. Values are given as ‘y’
for yes votes and ‘n’ for no votes. The training data is in politicians_train.tsv, and the test
data in politicians_test.tsv.

2. education: The second task is to predict the final grade (A, not A) for high school students. The
attributes (covariates, predictors) are student grades on 5 multiple choice assignments M1 through
M5, 4 programming assignments P1 through P4, and the final exam F. The training data is in
education_train.tsv, and the test data in education_test.tsv. 

3. Finally, there is a large dataset of mushroom samples, mushrooms_train.tsv
and mushrooms_test.tsv,  to test the algorithm. Each sample has discrete
attributes which were split into boolean attributes. For example, cap-size could be bell, conical,
convex, flat, knobbed, or sunken. This was split into boolean attributes cap-size bell, cap-size conical,
cap-size convex, etc. The goal is to differentiate the poisonous versus edible mushrooms.

#### Logistic Regression 

Problem Statement :
To implement a working Natural Language Processing (NLP) system, i.e.,
a sentiment polarity analyzer, using binary logistic regression. This would be used to determine
whether a review is positive or negative using movie reviews as data.

Feature Engineering : 
There are two types of feature engineering that has been done. 
1. Bag of words method
2. Word to Vec embeddings 

#### Neural Networks 
Problem Statement :

To implement an optical character recognizer using a one hidden layer neural network with sigmoid activations and a softmax output. 
It is used to classify the given set of black and white images into either of 4 classes: automobile, bird, frog, ship.
using a subset of a standard Computer Vision dataset, CIFAR-10 by implementing a single hidden layer neural network. 
It uses adagrad, a variant of the stochastic gradient descent to learn the parameters. 

#### Hidden Markov Model 

Problem Statement : 
To implement a named entity recognition system using Hidden Markov
Models (HMMs). Named entity recognition (NER) is the task of classifying named entities, typically proper
nouns, into pre-defined categories, such as person, location, organization. The WikiANN dataset has been used. 
The WikiANN dataset provides labelled entity data for Wikipedia articles in 282 languages. 
The English and French subset have primarily been used to predict the tags for a sequence of words. 


#### Deep Reinforcement Learning

Problem Statement: 

To implement Q-learning with linear function approximation to solve the
mountain car environment. It will implement all of the functions needed to initialize, train, evaluate,
and obtain the optimal policies and action values with Q-learning. In Mountain
Car we would control a car that starts at the bottom of a valley. The goal is to reach the flag at the top of the mountain.
However, the car is under-powered and can not climb up the hill by itself. Instead we
learn to leverage gravity and momentum to make our way to the flag. 
