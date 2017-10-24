#!/usr/bin/python

import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tester import dump_classifier_and_data
sys.path.append("../mini-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit  # noqa

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

# Will be using the totals from payment and stock compensations together with
# newly calculated ratios for email data in our model. For reasoning behind
# this continue to read here below.
features_list = ["poi",
                 # "total_payments",
                 # "total_stock_value",
                 'salary',
                 'deferral_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 "to_poi_ratio",
                 "from_poi_ratio",
                 "shared_with_poi_ratio"
                 # 'to_messages',
                 # 'from_poi_to_this_person',
                 # 'from_messages',
                 # 'from_this_person_to_poi',
                 # 'shared_receipt_with_poi'
                 ]

# Task 2: Remove outliers
# Task 3: Create new feature(s)

# Start by creating a dataframe from data_dict for easier analysis.
data_df = pd.DataFrame(data_dict).transpose()

# And drop the irrelevant email_address column.
data_df = data_df.drop("email_address", axis=1)

# After dropping the email_address feature all remaining features are
# numerical so we can easily convert the whole frame to numerical values.
data_df = data_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Create new features for the email data:
# - to_poi_ratio          (ratio of outgoing emails addressed to POI)
# - from_poi_ratio        (ratio of incoming emails from POI)
# - shared_with_poi_ratio (ratio of incoming emails which are shared with POI)
data_df["to_poi_ratio"] = (data_df["from_this_person_to_poi"]
                           / data_df["from_messages"])
data_df["from_poi_ratio"] = (data_df["from_poi_to_this_person"]
                             / data_df["to_messages"])
data_df["shared_with_poi_ratio"] = (data_df["shared_receipt_with_poi"]
                                    / data_df["to_messages"])
# Since we don't have all emails from all users and the number of total emails
# changes from person to person we convert the email features to ratios instead
# of absolute numbers.

# Prepare our plotting byt choosing style and defining columns to plot.
plt.style.use("ggplot")
email_ratio_cols = ["to_poi_ratio", "from_poi_ratio", "shared_with_poi_ratio"]
financial_data_cols = ['salary', 'deferral_payments',
                       'loan_advances', 'bonus', 'restricted_stock_deferred',
                       'deferred_income', 'expenses',
                       'exercised_stock_options', 'other',
                       'long_term_incentive', 'restricted_stock',
                       'director_fees']
financial_totals_cols = ["total_payments", "total_stock_value"]

# Plot our email ratios to look for outliers.
fig, ax = plt.subplots()
data_df[email_ratio_cols].plot(kind="box", ax=ax)
ax.set_title("Email ratios by feature")
plt.savefig("email_ratios_boxplot.jpg")
print("---")
print("Output email ratios box plot to email_ratios_boxplot.jpg")

# There are one data point in the to_poi_ratio feature with a 1.0 ratio.
print("---")
print("# Outlier from our to_poi_ratio data: ")
print(data_df[data_df["to_poi_ratio"] == 1.0][email_ratio_cols])
# After looking at the print out we can see that all the data point seems to
# be valid and will be left as is.

# Proceeding to plot the financial data
fig, ax = plt.subplots()
data_df[financial_data_cols].plot(kind="box",
                                  vert=False,
                                  figsize=(16, 9),
                                  ax=ax)
ax.set_title("Financial data by features")
plt.savefig("financial_data_boxplot.jpg")
print("---")
print("Output financial data box plot to financial_data_boxplot.jpg")

# There are some data points standing out with a value much higher than the
# other values in their category.
print("---")
print("Outliers from our financial data: ")
print([data_df.loc[data_df[x].idxmax()]
       for x in data_df[financial_data_cols].columns.values])
# After looking at the print out we can see that some data points have the
# index TOTAL. Dropping row with index TOTAL since this obviously isn't a
# person of interest.
print("TOTAL row for financial data will be dropped.")
data_df = data_df.drop("TOTAL")

# As a last note we can also see that among the column names there are to
# columns in the financial data that are total values of the other columns,
# total_payments and total_stock_value. A quick check in the description of the
# data found here we verifies this. To keep our model simple and keep down
# the number of unreported data points we will only use the totals of payments
# and stock value in our model at first.

# Let's plot some histograms of our five choosen features to see if they are
# relevant to our model or not.

# Starting by creating a new data frame for plotting email data.
email_df = data_df.pivot(index=data_df.index.values,
                         columns="poi")[email_ratio_cols]
# Getting the figure and axes
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
fig.suptitle("Email Ratio Histograms", fontsize=18)
# Loop through the columns we want to plot and add their subplots
i = 0
for col in email_ratio_cols:
    email_df[col].plot(kind="hist",
                       alpha=0.9,
                       title=col,
                       range=(0, 1),
                       figsize=(16, 5),
                       ax=ax[i])
    ax[i].set_yscale("log")
    i = i + 1
plt.savefig("email_ratio_histogram.jpg")
print("---")
print("Output email ratio histogram to email_ratio__histogram.jpg")

# And lastly for our financial totals
financials_df = data_df.pivot(index=data_df.index.values,
                              columns="poi")[financial_data_cols]
fig, ax = plt.subplots(nrows=2, ncols=6, sharey=True)
fig.suptitle("Financial Data Histograms", fontsize=18)
i = 0
for col in financial_data_cols:
    financials_df[col].plot(kind="hist",
                            alpha=0.9,
                            title=col,
                            figsize=(16, 7),
                            ax=ax[i / 6][i % 6])
    ax[i / 6][i % 6].set_yscale("log")
    i = i + 1
plt.savefig("financial_data_histogram.jpg")
print("---")
print("Output financial data histogram to financial_data_histogram.jpg")

# Financial features seems to skew slightly towards higher values for pois
# compared to non pois. For the email ratios there are no common trend but all
# features seems to have some difference in distribution between pois and non
# pois, I do have my doubts regarding the from from_poi_ratio. For know we will
# go on and use these five features until we have more evidence for or against
# any of them.


# As a last step we convert all our NaNs into strings to keep compatibility
# with the Udacity provided code.
# Below snippet from:
# https://discussions.udacity.com/t/featureformat-function-not-doing-its-job/192923
data_df = data_df.replace(np.nan, 'NaN', regex=True)

# Here we convert our data frame back to a data dictionary before we
# proceeding further.
data_dict = data_df.transpose().to_dict()

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Using the flow chart provided at
# http://scikit-learn.org/stable/tutor1ial/machine_learning_map/index.html for
# choosing the right estimator I decided to start by giving a linear SVC and
# KNeighbors classifier a comparison.
from sklearn.pipeline import Pipeline  # noqa
from sklearn.preprocessing import MinMaxScaler  # noqa
from sklearn.feature_selection import SelectKBest, chi2  # noqa
from sklearn.svm import LinearSVC, SVC  # noqa
from sklearn.neighbors import KNeighborsClassifier  # noqa
from sklearn.ensemble import RandomForestClassifier  # noqa
from sklearn import tree  # noqa
from sklearn.grid_search import GridSearchCV  # noqa
from sklearn.cross_validation import StratifiedShuffleSplit  # noqa

# Pipeline for a DecisionTree classifier
tree = Pipeline([
    ('feature_selection', SelectKBest()),
    ('classification', tree.DecisionTreeClassifier(class_weight="balanced",
                                                   random_state=42))
])
tree_param_grid = {"feature_selection__k": range(1, 16),
                   "classification__criterion": ["gini", "entropy"],
                   "classification__min_samples_split": range(2, 11)}

# Pipeline for a RandomForest classifier
forest = Pipeline([
    ('feature_selection', SelectKBest()),
    ('classification', RandomForestClassifier(class_weight="balanced",
                                              random_state=42))
])
forest_param_grid = {"feature_selection__k": range(1, 16),
                     "classification__n_estimators": [5, 10, 15, 20, 25],
                     "classification__criterion": ["gini", "entropy"],
                     "classification__min_samples_split": range(2, 11)}

# uncomment one of the below lines before running the script.
# clf = LinearSVC()
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = SVC(kernel="rbf")
# clf = AdaBoostClassifier(n_estimators=100)
# clf = tree.DecisionTreeClassifier(criterion="entropy",
#                                   min_samples_split=2,
#                                   random_state=42)

# LinearSVC results
# C=  1, Precision: 0.15760      Recall: 0.28400
# C= 10, Precision: 0.18195      Recall: 0.32350
# C= 50, Precision: 0.17712      Recall: 0.32750
# C=100, Precision: 0.20414      Recall: 0.34500
# C=200, Precision: 0.21149      Recall: 0.31100
# C=500, Precision: 0.22208      Recall: 0.26850

# KNeighbors results
# n_neighbors= 4, Precision: 0.24845      Recall: 0.02000
# n_neighbors= 5, Precision: 0.53774      Recall: 0.11400
# n_neighbors= 6, -
# n_neighbors= 7, Precision: 0.01460      Recall: 0.00100

# SVC(rbf) results
# C=100, gamma=1.0, -
# C=100, gamma=1.5, -

# Having a hard time to get any good results due to few data points try
# ensemble algorithms instead.

# AdaBoost results
# n_estimators= 50, learning_rate=1.0, Precision: 0.23992      Recall: 0.18150
# n_estimators=100, learning_rate=1.0, Precision: 0.24167      Recall: 0.18500

# Since many of the classifiers either have poor results due to the number of
# data points we start searching for a simpler model to solve our
# classification problem with. Naive Bayes could be and alternative but I am
# not confident in assuming that all our features are independent of each
# other, same goes for using logistic regression. In the end we start by trying
# a decision tree classifier.

# DecisionTreeClassifier results
# min_samples_split= 2, Precision: 0.27052      Recall: 0.24550
# min_samples_split= 3, Precision: 0.25823      Recall: 0.22750
# min_samples_split= 4, Precision: 0.23229      Recall: 0.20500
# Switching to entropy for impurity measures
# min_samples_split= 2, Precision: 0.33449      Recall: 0.28850
# min_samples_split= 3, Precision: 0.33631      Recall: 0.28250
# min_samples_split= 4, Precision: 0.29434      Recall: 0.27300

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split  # noqa

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

cv = StratifiedShuffleSplit(labels_train,
                            n_iter=100,
                            random_state=42)

tree = GridSearchCV(estimator=tree,
                    param_grid=tree_param_grid,
                    scoring="f1",
                    cv=cv)
tree = tree.fit(features_train, labels_train)
print("---")
print(tree.best_estimator_)
print("F1 score: ", tree.best_score_)

forest = GridSearchCV(estimator=forest,
                      param_grid=forest_param_grid,
                      scoring="f1",
                      cv=cv)
forest = forest.fit(features_train, labels_train)
print("---")
print(forest.best_estimator_)
print("F1 score: ", forest.best_score_)

if tree.best_score_ > forest.best_score_:
    clf = tree.best_estimator_
else:
    clf = forest.best_estimator_

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
