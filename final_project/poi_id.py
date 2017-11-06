#!/usr/bin/python

import sys
import pickle

# Imports for initial analysis and feature selection.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports for the machine learning models.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Imports for model tuning and evaluation.
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

# Imports for helpers provided via the Udacity sample code.
from tester import dump_classifier_and_data, test_classifier
sys.path.append("../mini-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit  # noqa

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# For this analysis we will be using the totals from payment and stock
# compensations together with newly calculated ratios for the email data.
# For the reasoning behind these choices read here below.
features_list = ["poi",
                 "restricted_stock_deferred",
                 "deferral_payments",
                 "director_fees",
                 "from_poi_ratio",
                 "other",
                 "expenses",
                 "loan_advances",
                 "restricted_stock",
                 "shared_with_poi_ratio",
                 "long_term_incentive",
                 "deferred_income",
                 "to_poi_ratio",
                 "salary",
                 "bonus",
                 "exercised_stock_options"
                 ]


# Preparation 1: Create pandas dataframe.

# Start by creating a dataframe from data_dict for easier analysis.
data_df = pd.DataFrame(data_dict).transpose()

# Drop the superfluous email_address column together with the totals columns in
# in the financial data.
data_df = data_df.drop("email_address", axis=1)
data_df = data_df.drop("total_payments", axis=1)
data_df = data_df.drop("total_stock_value", axis=1)

# All remaining features are numerical so we can easily convert the whole frame
# to numerical values.
data_df = data_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Prepare our plotting byt choosing style and defining columns to plot.
plt.style.use("ggplot")
financial_data_cols = ['salary', 'deferral_payments',
                       'loan_advances', 'bonus', 'restricted_stock_deferred',
                       'deferred_income', 'expenses',
                       'exercised_stock_options', 'other',
                       'long_term_incentive', 'restricted_stock',
                       'director_fees']
email_data_cols = ["to_messages", "from_messages", "from_poi_to_this_person",
                   "from_this_person_to_poi", "shared_receipt_with_poi"]
email_ratio_cols = ["to_poi_ratio", "from_poi_ratio", "shared_with_poi_ratio"]

# Task 2: Remove outliers

# Plotting the financial data
fig, ax = plt.subplots()
data_df[financial_data_cols].plot(kind="box",
                                  vert=False,
                                  figsize=(16, 9),
                                  ax=ax)
ax.set_title("Financial data by features")
ax.set_xlabel("Amount in dollars")
plt.savefig("financial_data_boxplot.jpg")
print "---"  # noqa
print "Output financial data box plot to financial_data_boxplot.jpg"

# There are some data points standing out with a value much higher than the
# other values in their category. Printing out salary as an example.
print "---"
print "Outlier in financial data found: "
print data_df.loc[data_df["salary"].idxmax()]

# After looking at the print out we can see that the index is TOTAL.
# Dropping this row since this obviously isn't a person of interest.
print "TOTAL rows for financial data will be dropped."
data_df = data_df.drop("TOTAL")

# While checking the pdf file with the table containing the original data I also
# spotted another entry that was out of place. Since most indexes are on the
# form <Last Name> <First Name> or <Last Name> <First Name> <Initial> lets have
# a closer look at indexes having less than two or more than three words in
# them.
print "Indexes with less than two or more than three words: "
print [x for x in data_df.index if len(x.split(" ")) > 3]
print "THE TRAVEL AGENCY IN THE PARK outlier found."
print "THE TRAVEL AGENCY IN THE PARK row will be dropped."
data_df = data_df.drop("THE TRAVEL AGENCY IN THE PARK")

# Plotting the email data
fig, ax = plt.subplots()
data_df[email_data_cols].plot(kind="box",
                              vert=False,
                              figsize=(16, 9),
                              ax=ax)
ax.set_title("Email data by features")
ax.set_xlabel("Number of emails")
plt.savefig("email_data_boxplot.jpg")
print "---"
print "Output email data box plot to email_data_boxplot.jpg"
# As seen in the plot and given the same procedure as from the financial data
# there are some data points that sticks out as having remarkably high counts
# but after further inspection they seem valid.


# Task 3: Create new feature(s)

# Since we don't have all emails from all users and the number of total emails
# changes from person to person we convert the email features to ratios instead
# of absolute numbers.
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

# Preparation 2: Descriptive statistics and final feature evaluation

print "---"
print "Printing descriptive statistics for the POI label:"
print data_df["poi"].describe()
# We can see here that we have a total of 145 data points, 127 labeled as
# non-POI and only 18 POIs which accounts for only 12%. Many algorithms expect
# a lot more balanced ratio between classes in this type of classification
# problems so we need to take this into account when choosing our machine
# learning algorithm later on. No unknowns are listed.
# Worth to note here is that with this much unbalance between POIs and non-POIs
# by just guessing non-POI we will achieve an accuracy of 88%. Due to this we
# will use the F score together with recall and precision while evaluating our
# algorithms rather than just accuracy.

print "---"
print "Printing descriptive statistics for the email ratio features:"
print data_df[email_ratio_cols].describe()
# For the email data we can see that we have 86 data points, around 60% of the
# total dataset. Since we recalculated the features to ratios all values are
# lying in the 0 to 1 interval.

print "---"
print "Printing descriptive statistics for the financial features:"
print data_df[financial_data_cols].describe()
# The financial data isn't as uniform as either the poi or email ratios with
# different counts, values spanning both positive and negative numbers, largely
# different sizes for different features etc.


# Let's plot some histograms of our chosen features to see if they are
# relevant to our model or not.

# Starting by creating a new data frame for plotting email data.
email_df = data_df.pivot(index=data_df.index.values,
                         columns="poi")[email_ratio_cols]
# Getting the figure and axes
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
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
print "---"
print "Output email ratio histogram to email_ratio__histogram.jpg"

# And now for our financial data:
financials_df = data_df.pivot(index=data_df.index.values,
                              columns="poi")[financial_data_cols]
fig, ax = plt.subplots(nrows=2, ncols=6, sharey=True)
fig.tight_layout()
i = 0
for col in financial_data_cols:
    financials_df[col].plot(kind="hist",
                            alpha=0.9,
                            title=col,
                            figsize=(16, 7),
                            ax=ax[i / 6][i % 6])
    ax[i / 6][i % 6].set_yscale("log")
    ax[i / 6][i % 6].get_xaxis().get_major_formatter().set_powerlimits((-3, 4))
    i = i + 1
plt.savefig("financial_data_histogram.jpg")
print "---"
print "Output financial data histogram to financial_data_histogram.jpg"

# Financial features seems to skew slightly towards higher values for POIs
# compared to non POIs. For the email ratios there are no common trend but all
# features seems to have some difference in distribution between POIs and non
# POIs, I do have my doubts regarding the from from_poi_ratio. For know we will
# go on and use these five features until we have more evidence for or against
# any of them.

# Before training we will use SelectKBest to calculate ANOVA
# F-statistic for each feature we have chosen. The idea comes from
# this forum thread
# (https://discussions.udacity.com/t/deciding-on-how-many-features-i-should-use/160726)
# and the poor performance I had on my laptop trying to compute SelectKBest
# selections directly in the Pipeline for all ks.
k_best = SelectKBest(k="all")
feature_cols = email_ratio_cols + financial_data_cols
best_features = k_best.fit(X=data_df[feature_cols].fillna(0),
                           y=data_df["poi"].fillna(0))
feature_scores = pd.Series(data=best_features.scores_,
                           index=feature_cols)
print "---"
print "F-values for each feature:"
print feature_scores.sort_values()
# There are some small gaps in the sorted values of F-scores, we will use each
# such gap a candidate for a k-value. (e.g. k=[2, 5, 7, 11, 13, 15]
# restricted_stock_deferred     0.064477
# deferral_payments             0.209706
# director_fees                 2.089310
# from_poi_ratio                3.293829
# other                         4.263577
# expenses                      6.374614
# loan_advances                 7.301407
# restricted_stock              9.480743
# shared_with_poi_ratio         9.491458
# long_term_incentive          10.222904
# deferred_income              11.732698
# to_poi_ratio                 16.873870
# salary                       18.861795
# bonus                        21.327890
# exercised_stock_options      25.380105


# Preparation 3: Convert the dataframe back to previous datadict format

# As a last step before choosing our model we convert all our NaNs into strings
# to keep compatibility with the Udacity provided code.
# Below snippet from:
# https://discussions.udacity.com/t/featureformat-function-not-doing-its-job/192923
data_df = data_df.replace(np.nan, 'NaN', regex=True)

# We convert our dataframe back to a data dictionary before proceeding further.
data_dict = data_df.transpose().to_dict()

# Store to my_dataset for easy export below.
my_dataset = data_dict


# Task 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# We start by pitting a SVM against a decision tree to see which algorithm is
# better suited.

# Pipeline for a SVC(SVM) classifier
svc = Pipeline([
    ("feature_scaling", StandardScaler()),
    ("feature_selection", SelectKBest()),
    ("classification", SVC(kernel="rbf",
                           class_weight="balanced",
                           random_state=42))
])
svc_param_grid = {"feature_selection__k": [2, 5, 7, 11, 13, 15],
                  "classification__C": [0.1, 1, 10, 100],
                  "classification__gamma": [0.01, 0.1, 1, 10]}

# Pipeline for a DecisionTree classifier
tree = Pipeline([
    ("feature_selection", SelectKBest()),
    ("classification", DecisionTreeClassifier(class_weight="balanced",
                                              random_state=42))
])
tree_param_grid = {"feature_selection__k": [2, 5, 7, 11, 13, 15],
                   "classification__criterion": ["gini", "entropy"],
                   "classification__min_samples_split": range(2, 21)}

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Using StratifiedShuffleSplit to create splits for our GridSearchCV function
# in order to preserve the ratio between non-POIs and POIs in our small data
# set. Also, the same algorithm is used in the Udacity text script meaning that
# we will get results more representative for the final evaluation using the
# same method.
cv = StratifiedShuffleSplit(labels_train,
                            n_iter=1000,
                            random_state=42)

svc = GridSearchCV(estimator=svc,
                   param_grid=svc_param_grid,
                   scoring="f1",
                   cv=cv)
svc = svc.fit(features_train, labels_train)
print "---"
print "# SVM performance for best parameter set"
test_classifier(svc.best_estimator_, my_dataset, features_list)

tree = GridSearchCV(estimator=tree,
                    param_grid=tree_param_grid,
                    scoring="f1",
                    cv=cv)
tree = tree.fit(features_train, labels_train)
print "---"
print "# Decision Tree performance for best parameter set"
test_classifier(tree.best_estimator_, my_dataset, features_list)

# As mentioned in the beginning let's go back and test our hypothesis regarding
# that the newly created email ratios we created gives better performance than
# the originally provided email features by running the Decision Tree
# classifier one more time but with the alternative feature set.
# Extract features and labels from dataset for local testing
alt_features_list = [x for x in features_list if x not in email_ratio_cols]
alt_features_list.extend(email_data_cols)
alt_data = featureFormat(my_dataset, alt_features_list, sort_keys=True)
alt_labels, alt_features = targetFeatureSplit(alt_data)
alt_features_train, alt_features_test, alt_labels_train, alt_labels_test = \
    train_test_split(alt_features, alt_labels, test_size=0.3, random_state=42)

alt_tree = tree.fit(alt_features_train, alt_labels_train)
print "---"
print """
# Decision Tree performance for best parameter set with alternative features.
"""
test_classifier(alt_tree.best_estimator_, my_dataset, alt_features_list)
# As we can see from the above code's print out we get a higher F1 score from
# using our ratios. We will continue forward using them as input for our model.

# Using the Decision Tree algorithm for classification seems promising. But I
# suspect we still can get better results. To do this let's try Random Forest
# algorithm instead where we generate multiple trees and combine their results
# in order to classify a data point. This could even out some of the issues
# that might be due to overfitting to our training data set. We will use the
# same parameters as our best tree and only change the number of trees
# generated.

# Pipeline for a Random Forest classifier
forest = Pipeline([
    ("feature_selection", SelectKBest()),
    ("classification", RandomForestClassifier(class_weight="balanced",
                                              random_state=42))
])
forest_param_grid = {"feature_selection__k": [7],
                     "classification__criterion": ["gini"],
                     "classification__min_samples_split": [19],
                     "classification__n_estimators": [2, 5, 10, 25, 50, 100,
                                                      150]}
forest = GridSearchCV(estimator=forest,
                      param_grid=forest_param_grid,
                      scoring="f1",
                      cv=cv)

forest = forest.fit(features_train, labels_train)
print "---"
print "# Random Forest performance for best parameter set:"
test_classifier(forest.best_estimator_, my_dataset, features_list)
print "Chosen features for the estimator: "
selected_features = \
        forest.best_estimator_.named_steps["feature_selection"].get_support(
            indices=True
        )
print ", ".join([features_list[x] for x in selected_features])

# Plot scores for each estimator value.
estimators = [x[0]["classification__n_estimators"]
              for x in forest.grid_scores_]
scores = [x[1] for x in forest.grid_scores_]
fig, ax = plt.subplots()
ax.plot(estimators, scores)
ax.set_title("F1-score vs number of estimators in Random Forest Model")
ax.set_xlabel("F1-Score")
ax.set_ylabel("Number of estimators")
plt.savefig("f1_vs_estimators.jpg")
print "---"
print "Output F1-score vs estimators plot to f1_vs_estimators.jpg"
# We can see that the starts to overfit and get worse results for values of
# n_estimators after 100 so we set the parameter to 100.

# Initializing the final best classifier with all parameters and exporting it
# as clf.
clf = Pipeline([
    ("feature_selection", SelectKBest(k=7)),
    ("classification", RandomForestClassifier(criterion="gini",
                                              min_samples_split=19,
                                              n_estimators=100,
                                              class_weight="balanced",
                                              random_state=42))
])

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
