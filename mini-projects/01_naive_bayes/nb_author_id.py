#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Train the Gaussian model on our email data set.
t0 = time()
classifier = GaussianNB()
classifier.fit(features_train, labels_train)
print "Training time (fit): ", round(time()-t0, 3), "s"
# Training time (fit):  1.483 s

# Create predition data to measure accurancy
t1 = time()
pred = classifier.predict(features_test)
print "Training time (predict): ", round(time()-t1, 3), "s"
# Training time (predict):  0.155 s

# Check accuracy for the resulting model
print accuracy_score(pred, labels_test)
# Result: 0.973265073948

#########################################################


