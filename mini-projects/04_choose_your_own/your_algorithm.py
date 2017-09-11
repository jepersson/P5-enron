#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


print ""
print "### Previous Results ###"

# Train the Random Forest on our email data set.
t0 = time()
clf = RandomForestClassifier(n_estimators=100,random_state=0,
                             min_samples_leaf=40)
clf = clf.fit(features_train, labels_train)
print "Training time (fit): ", round(time()-t0, 3), "s"
# Training time (fit):  

# Create predition data to measure accurancy
t1 = time()
pred = clf.predict(features_test)
print "Training time (predict): ", round(time()-t1, 3), "s"
# Training time (predict): 

# Check accuracy for the resulting model
print "Accuracy: ", accuracy_score(pred, labels_test)
# Result: 

print ""
print "### New Results ###"

# Train the Random Forest on our email data set.
t0 = time()
clf = RandomForestClassifier(n_estimators=10000,random_state=0,
                             min_samples_leaf=25)
clf = clf.fit(features_train, labels_train)
print "Training time (fit): ", round(time()-t0, 3), "s"
# Training time (fit):  

# Create predition data to measure accurancy
t1 = time()
pred = clf.predict(features_test)
print "Training time (predict): ", round(time()-t1, 3), "s"
# Training time (predict): 

# Check accuracy for the resulting model
print "Accuracy: ", accuracy_score(pred, labels_test)
# Result: 
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

plt.show()
