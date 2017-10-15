#!/usr/bin/python

import sys
import pickle

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
features_list = ['poi', 'salary']  # You will need to use more features

# Task 2: Remove outliers
# Task 3: Create new feature(s)

# Start by creating a dataframe from data_dict for easier feature analysis.
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

# Prepare our plotting byt choosing style and columns to plot.
plt.style.use("ggplot")
email_ratio_cols = ["to_poi_ratio", "from_poi_ratio", "shared_with_poi_ratio"]
email_data_cols = ["from_this_person_to_poi", "from_poi_to_this_person",
                   "to_messages", "from_messages", "shared_receipt_with_poi"]
financial_data_cols = ['salary', 'deferral_payments', 'total_payments',
                       'loan_advances', 'bonus', 'restricted_stock_deferred',
                       'deferred_income', 'total_stock_value', 'expenses',
                       'exercised_stock_options', 'other',
                       'long_term_incentive', 'restricted_stock',
                       'director_fees']

# Plot our email data to look for outliers.
fig, ax = plt.subplots()
data_df[email_data_cols].plot(kind="box",
                              ax=ax)
ax.set_title("Email data by features")
plt.show()

# We have two to_messages and one from_messages data points that are exceeding
# 12000 counts.
print("===")
print("Outliers from our to_messages data: ")
print(data_df[data_df["to_messages"] > 12000][email_data_cols])
print("Outliers from our from_messages data: ")
print(data_df[data_df["from_messages"] > 12000][email_data_cols])
print("===")
# After looking at the print out we can see that all the data points seems to
# be valid and they will be left as is.

# Plot our email ratios to look for outliers.
fig, ax = plt.subplots()
data_df[email_ratio_cols].plot(kind="box",
                               ax=ax)
ax.set_title("Email ratios by feature")
plt.show()

# There are one data point in the to_poi_ratio feature with a 1.0 ratio.
print("===")
print("Outlier from our to_poi_ratio data: ")
print(data_df[data_df["to_poi_ratio"] == 1.0][email_data_cols])
print("===")
# After looking at the print out we can see that all the data point seems to
# be valid and will be left as is.

# Proceeding to plot the financial data
fig, ax = plt.subplots()
data_df[financial_data_cols].plot(kind="box",
                                  vert=False,
                                  ax=ax)
ax.set_title("Financial data by features")
plt.show()

# There are nine data points standing out with a value much higher than the
# other values in their category.
print("===")
print("Outliers from our financial data: ")
print([data_df.loc[data_df[x].idxmax()]
       for x in data_df.columns.values])
print("===")
# After looking at the print out we can see that some data points have the
# index TOTAL. Adding code to filter out the TOTAL entry.

# Drop row with index TOTAL since this obviously isn't a person of
# interest.
data_df = data_df.drop("TOTAL")
# As a last note we can also see that among the column names there are to
# columns in the financial data that seems to be sums of the other columns,
# total_payments and total_stock_value. A quick check in the description of the
# data found here we can verify this. To keep our model simple and keep down
# the number of unreported data points we will only use the totals of payments
# and stock value in our model at first.

# Here we convert back our data frame back to a data dictionary before we
# proceed further.
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
from sklearn.naive_bayes import GaussianNB  # noqa
clf = GaussianNB()

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


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
