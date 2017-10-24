Step 1: Cleaning up mail features

First I took a look at the different features for the email data. 
Since there might be a difference between users in what way and volume emails
are being sent I decided to create new features based on the ratio between total
amount of sent and received email and the amount of emails sent and received
to/from POI's. Also, the email address feature are most likely not related to
if the person is a POI or not so this will not be used in the model at all.

Code for creating new email features:

```python
data_df["to_poi_ratio"] = (data_df["from_this_person_to_poi"]
                           / data_df["from_messages"])
data_df["from_poi_ratio"] = (data_df["from_poi_to_this_person"]
                             / data_df["to_messages"])
data_df["shared_with_poi_ratio"] = (data_df["shared_receipt_with_poi"]
                                    / data_df["to_messages"])
```
Incidentially we also find an error in our data whre one person recieved more
emails with an shared reciept with POI than emails in total. The error is
marginal and no good candidate for what value to substitue.

After creating the new features using the following lines to create a boxplot
with matplotlib and verify outliers.

```python


```

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                      weights='uniform')
                              Accuracy: 0.86880       Precision: 0.53774
                              Recall: 0.11400 F1: 0.18812     F2: 0.13533
                                      Total predictions: 15000        True
                                      positives:  228    False positives:  196
                                      False negatives: 1772   True negatives:
                                      12804

LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
               verbose=0)
                       Accuracy: 0.71393       Precision: 0.15979      Recall:
                       0.26900 F1: 0.20048     F2: 0.23665
                               Total predictions: 15000        True positives:
                               538    False positives: 2829   False negatives:
                               1462   True negatives: 10171


