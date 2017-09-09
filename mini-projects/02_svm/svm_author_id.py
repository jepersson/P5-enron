#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# your code goes here #

from sklearn import svm
from sklearn.metrics import accuracy_score

# Train the SVM on our email data set.
t0 = time()
clf = svm.SVC(kernel="linear")
clf.fit(features_train, labels_train)
print "Training time (fit): ", round(time()-t0, 3), "s"
# Training time (fit):  175.162 s

# Create predition data to measure accurancy
t1 = time()
pred = clf.predict(features_test)
print "Training time (predict): ", round(time()-t1, 3), "s"
# Training time (predict):  18.653 s

# Check accuracy for the resulting model
print accuracy_score(pred, labels_test)
# Result: 0.984072810011

# Try once more but with just 1% of the training data set
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

# Train the SVM on our smaller email data set.
t0 = time()
clf = svm.SVC(kernel="linear")
clf.fit(features_train, labels_train)
print "Training time (fit, smaller data set): ", round(time()-t0, 3), "s"
# Training time (fit, smaller data set): 0.119 s

# Create predition data to measure accurancy
t1 = time()
pred = clf.predict(features_test)
print "Training time (predict, smaller data set): ", round(time()-t1, 3), "s"
# Training time (predict, smaller data set): 1.107 s

# Check accuracy for the resulting model trained with smaller data set
print accuracy_score(pred, labels_test)
# Result: 0.884527872582

# Try something else by switching the kernel used for the smaller data set
t0 = time()
clf = svm.SVC(kernel="rbf")
clf.fit(features_train, labels_train)
print "Training time (fit, smaller data set, RBF): ", round(time()-t0, 3), "s"
# Training time (fit, smaller data set, RBF): 0.127 s

# Create predition data to measure accurancy
t1 = time()
pred = clf.predict(features_test)
print "Training time (predict, smaller data set, RBF): ", round(time()-t1, 3), "s"
# Training time (predict, smaller data set, RBF): 1.301 s

# Check accuracy for the resulting model trained with smaller data set
print accuracy_score(pred, labels_test)
# Result: 0.616040955631

#  Start playing around with C parameter to see what happens

# C = 10.0
clf = svm.SVC(C=10.0, kernel="rbf")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "Accuracy score (smaller data set, RBF-kernel, C = 10): ", acc
# Result: 0.616040955631

# C = 100.0
clf = svm.SVC(C=100.0, kernel="rbf")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "Accuracy score (smaller data set, RBF-kernel, C = 100): ", acc
# Result: 0.616040955631

# C = 1000.0
clf = svm.SVC(C=1000.0, kernel="rbf")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "Accuracy score (smaller data set, RBF-kernel, C = 1000): ", acc
# Result: 0.821387940842

# C = 10000.0
clf = svm.SVC(C=10000.0, kernel="rbf")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "Accuracy score (smaller data set, RBF-kernel, C = 10000): ", acc
# Result: 0.892491467577

# Reload the full data set once again
features_train, features_test, labels_train, labels_test = preprocess()

# And try running the C = 10000.0 code from above once again
# C = 10000.0, now with full training data set
clf = svm.SVC(C=10000.0, kernel="rbf")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "Accuracy score (smaller data set, RBF-kernel, C = 10000): ", acc
# Result: 0.990898748578

# 99% accuracy is pretty good. Now let's see what the model's predictions are
# for data point 10, 26, and 50
print "Prediction (10): ", pred[10]
print "Prediction (26): ", pred[26]
print "Prediction (50): ", pred[50]

# How many events are predicted to be Chris?
print "Predictions for Chris: ", len(pred[pred == 1])
print "Predictions for Sara: ", len(pred[pred == 0])
