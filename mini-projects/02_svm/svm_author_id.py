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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

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

#########################################################


