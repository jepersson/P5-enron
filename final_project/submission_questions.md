# Enron Submission Free-Response Questions

A critical part of machine learning is making sense of your analysis process and
communicating it to others. The questions below will help us understand your
decision-making process and allow us to give feedback on your project. Please
answer each question; your answers should be about 1-2 paragraphs per question.
If you find yourself writing much more than that, take a step back and see if
you can simplify your response!

When your evaluator looks at your responses, he or she will use a specific list
of rubric items to assess your answers. Here is the link to that rubric: [Link]
Each question has one or more specific rubric items associated with it, so
before you submit an answer, take a look at that part of the rubric. If your
response does not meet expectations for all rubric points, you will be asked to
revise and resubmit your project. Make sure that your responses are detailed
enough that the evaluator will be able to understand the steps you took and your
thought processes as you went through the data analysis.

Once you’ve submitted your responses, your coach will take a look and may ask a
few more focused follow-up questions on one or more of your answers.  

We can’t wait to see what you’ve put together for this project!

1. Summarize for us the goal of this project and how machine learning is useful
   in trying to accomplish it. As part of your answer, give some background on
   the dataset and how it can be used to answer the project question. Were there
   any outliers in the data when you got it, and how did you handle those?
   [relevant rubric items: “data exploration”, “outlier investigation”] 

The goal is to classify persons of interest(POI) present in a data set made
available during the Enron trial. The original data is consisting of a
collection of internal emails connected to the persons being on trial. The
version of the data set used in this project has been enhanced with financial
data also made available at the time. The target variable POI is a boolean where
people who evidently had a connection with the fraud which led to Enron's downfall
(e.g. a person found guilty, that settled, or witnessed in exchange for immunity) are
flagged.

Initially looking at the provided feature I made the decision to omit the email
feature right away since the email address most likely does not have any
connection to the probability of someone being a POI or not. The totals
(total_stock_value and total_payments) from the financial data were also
omitted since this information is redundant and possible to infer from the other
financial features. After some further investigation, a row of the dataset
containing totals (TOTAL) was also found and dropped from the financial data.
All other entries were left as is.  

2. What features did you end up using in your POI identifier, and what selection
   process did you use to pick them? Did you have to do any scaling? Why or why
   not? As part of the assignment, you should attempt to engineer your own
   feature that does not come ready-made in the dataset -- explain what feature
   you tried to make, and the rationale behind it. (You do not necessarily have
   to use it in the final analysis, only engineer and test it.) In your feature
   selection step, if you used an algorithm like a decision tree, please also
   give the feature importances of the features that you use, and if you used an
   automated feature selection function like SelectKBest, please report the
   feature scores and reasons for your choice of parameter values.  [relevant
   rubric items: “create new features”, “intelligently select features”,
   “properly scale features”]

After cleaning up the financial features attention was given to the email
features for the next step. The email data was spotty and we are unable to
discern if the emails included are exhaustive or not. To mitigate the effects
of possible selection bias due to shifting sample sizes the five existing email
features were combined into three ratios based on the amount of emails sent
to/from each person respectively. My hope is that doing this will help avoid
overfitting our model by reducing noise solely from the sampling method and we
will revisit this decision once more towards the end of the analysis to see if
my hypothesis is correct or not. The ratios were calculated as seen here below.

```python
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
```

After the initial clean up described above were finished a descriptive data
analysis were performed. We found that from a total of 145 data points 127 were
labeled non-POIs and 18 labeled as POIs. In total, we have 15 features at the
moment, not including the given email features since we will switch out these for
our new email ratio features instead. The newly made email features consists of
86 data points all of them spanning between 0 and 1 since we created them as
ratios. The most heterogeneous features come from the financial data features
which range from just 16 up to 109 data points depending on which features we
are looking at. However, the financial features differ from our email features
since non-existing values simply are non-existent (zero) while for the email
features we are unable to tell if no data really means no occurrences or if it
just means no data. Further, a list up of the F-values for each of the
remaining 15 features with respect to the POI label (All existing data given
were used for this calculation). Loosely based on the values distribution the
below values were chosen as input for a SelectKBest function in the later model
tuning step were they will be further evaluated using the exhaustive
GridSearchCV evaluation function.  

```
# F-values calculated for features with respect to POI.
restricted_stock_deferred     0.064477
deferral_payments             0.209706
director_fees                 2.089310
from_poi_ratio                3.293829
other                         4.263577
expenses                      6.374614
loan_advances                 7.301407
restricted_stock              9.480743
shared_with_poi_ratio         9.491458
long_term_incentive          10.222904
deferred_income              11.732698
to_poi_ratio                 16.873870
salary                       18.861795
bonus                        21.327890
exercised_stock_options      25.380105
```

Potential parameter Values chosen for SelectKBest's k:
```
[2, 5, 7, 11, 13, 15]
```

3. What algorithm did you end up using? What other one(s) did you try? How did
   model performance differ between algorithms?  [relevant rubric item: “pick an
   algorithm”]

Reflecting on the models for supervised learning being presented in the intro to
machine learning lesson material I wanted to choose a model that is as simple as
possible so that I still am able to make informed decisions when advancing into
the parameter tuning stage. This might say more about my current experience with
machine learning rather than the data itself but are nonetheless a valid point
to make.  This assumption gave me three candidates presented early in the
course: Naive Bayes, SVM, and Decision Trees. Naive Bayes were omitted since I
didn't feel confident about finding a good prior for our data without resorting
to pure guessing. This left the SVM and Decision Tree models as valid selections
for the next step.

One SVM and one Decision Tree classifier where compared and the Decision Tree
were chosen based on better performance in respect to its F1 value. The reason
for choosing the F1 value as a key indicator rather than accuracy is due to the
unbalanced nature of the POI label. There are only 18 POIs and 127 non-POIs in
our data set which means that we able to get a rather high accuracy (88%) just by
guessing non-POI for everything which isn't very helpful since our goal is to
classify potential POIs. Further, when comparing the models one extra step was
added to the SVM pipeline to standardize the input values since the SVM model
expects standardized features while the Decision Tree works better with
non-standardized features.

In the end, a RandomForest model was chosen to mitigate the risk of overfitting
our model and to increase the performance on our testing data set.

4. What does it mean to tune the parameters of an algorithm, and what can happen
   if you don’t do this well?  How did you tune the parameters of your
   particular algorithm? What parameters did you tune? (Some algorithms do not
   have parameters that you need to tune -- if this is the case for the one you
   picked, identify and briefly explain how you would have done it for the model
   that was not your final choice or a different model that does utilize
   parameter tuning, e.g. a decision tree classifier).  [relevant rubric items:
   “discuss parameter tuning”, “tune the algorithm”]

Parameter tuning is the process of finding the optimal or good enough values for
the input parameters the chosen model needs, excluding the target data set's
features and labels. The method I used to initially tune the parameters for the
SVM and Decision Tree models are called GridSearchCV and are provided in the
sklearn library. GridSearchCV performs an exhaustive search over all
combinations of the provided candidate parameter inputted and evaluates them
using a specified measure, in this case, the F1-value. The candidate values
that were chosen and performance for the models can be seen here below.

```python
# Candidate parameters for the SVC(SVM) model used in GridSearchCV.
svc_param_grid = {"feature_selection__k": [2, 5, 7, 11, 13, 15],
                  "classification__C": [0.1, 1, 10, 100],
                  "classification__gamma": [0.01, 0.1, 1, 10]}


# Candidate parameters for the Decision Tree model used in GridSearchCV.
tree_param_grid = {"feature_selection__k": [2, 5, 7, 11, 13, 15],
                   "classification__criterion": ["gini", "entropy"],
                   "classification__min_samples_split": range(2, 21)}
```

```
# SVM performance for best parameter set
Pipeline(memory=None,
         steps=[('feature_scaling', StandardScaler(copy=True, 
                                                   with_mean=True, 
                                                   with_std=True)), 
                ('feature_selection', SelectKBest(k=11, 
                                                  score_func=<function f_classif at 0x107cb4b90>)), 
                ('classification', SVC(C=1, 
                                       cache_size=200, 
                                       class_weight='balanced', 
                                       coef0=0.0,
                                       decision_function_shape='ovr', 
                                       degree=3, 
                                       gamma=0.01, 
                                       kernel='rbf',
                                       max_iter=-1, 
                                       probability=False, 
                                       random_state=42, 
                                       shrinking=True,
                                       tol=0.001, 
                                       verbose=False))])

Accuracy:   0.69007	
Precision:  0.24834     Recall: 0.65350	
F1:         0.35991     F2:     0.49272

Total predictions:  15000
True positives:     1307    False positives:    3956
False negatives:    693     True negatives:     9044

# Decision Tree performance for best parameter set
Pipeline(memory=None,
         steps=[('feature_selection', SelectKBest(k=7, 
                                                  score_func=<function f_classif at 0x107cb4b90>)), 
                ('classification', DecisionTreeClassifier(class_weight='balanced', 
                                                          criterion='gini',
                                                          max_depth=None, 
                                                          max_features=None, 
                                                          max_leaf_nodes=None,
                                                          min_impurity_decrease=0.0, 
                                                          min_impurity_split=None,
                                                          min_samples_leaf=1, 
                                                          min_samples_split=19,
                                                          min_weight_fraction_leaf=0.0, 
                                                          presort=False, 
                                                          random_state=42,
                                                          splitter='best'))])

Accuracy:   0.75687
Precision:  0.30104     Recall: 0.62300	
F1:         0.40593     F2:     0.51322

Total predictions:  15000
True positives:     1246        False positives:    2893	
False negatives:    754         True negatives:     10107

# Decision Tree performance for best parameter set with alternative features.
Pipeline(memory=None,
         steps=[('feature_selection', SelectKBest(k=15, 
                                                  score_func=<function f_classif at 0x10a528c08>)), 
                ('classification', DecisionTreeClassifier(class_weight='balanced', 
                                                          criterion='gini',
                                                          max_depth=None,
                                                          max_features=None,
                                                          max_leaf_nodes=None,
                                                          min_impurity_decrease=0.0,
                                                          min_impurity_split=None,
                                                          min_samples_leaf=1,
                                                          min_samples_split=11,
                                                          min_weight_fraction_leaf=0.0,
                                                          presort=False,
                                                          random_state=42,
                                                          splitter='best'))])

Accuracy:       0.76193
Precision:      0.23127     Recall:     0.33800 
F1:             0.27463     F2:         0.30944

Total predictions:  15000
True positives:     676     False positives:    2247
False negatives:    1324    True negatives:     10753

```

Given the above candidate hyperparameters, the Decision Tree model came out as a
winner. To validate our earlier hypothesis that our newly created ratios for the 
email features help improve predictions the Decision Tree model was fitted a 
second time over with an alternative feature set containing the original email data.
As we can see switching out the original email features in favor for the new ratios
increases our performance so we can use our own created features with confidence.
However, since using this model just like this is prone to overfitting
and not very common in the real world (or so I have heard) we didn't stop here.
Instead, we took our modeling one step further and used the tuned Decision Tree
hyperparameters as input for a Random Forest classifier which creates multiple
trees from subsets of our data to make many weak predictions that we then can
combine using majority voting to increase the stability of our model. Tuning
were performed once again with the below candidate parameters and the F1-score
for the chosen n_estimator values and by looking at the below plot we could
discern that the best value for our n_estimator is 100.

```python
# Candidate parameters fir the Random Forest model used in GridSearchCV
forest_param_grid = {"feature_selection__k": [7],
                     "classification__criterion": ["gini"],
                     "classification__min_samples_split": [19],
                     "classification__n_estimators": [2, 5, 10, 25, 50, 100,
                                                      150]}
```

![F1 scores vs n_estimators](/f1_vs_estimators.jpg)

5. What is validation, and what’s a classic mistake you can make if you do it
   wrong? How did you validate your analysis?  [relevant rubric items: “discuss
   validation”, “validation strategy”]

In machine learning validation means to check your model's general performance 
in contrast to its performance on the data used to fit the model initially.
This is to simulate how the model would perform using new and unseen data and is an
invaluable tool for avoiding overfitting your model to noise only existing in
the data used initially. Ideally, this means that we split up the data in three sets
one training set, one test set, and one validation set. Also, since the way the data
is split might affect how the outcome of the valuation repeating this the evaluation
over many different splits is also desirable.

This time in order to do this in practise we need to split up our data into different
sets, one used for training, one used for testing, and one for validation. But, since
our data set is rather small we will use our test set for both testing and
validation, and as mentioned earlier are labels are unbalanced
StratifiedShuffleSplit (also used in the provided tester.py) from sklearn where
used to split up our data set into testing and training sets while trying to
keep the ratio of POIs and non-POIs from the initial data. To make our
validation more robust we split the data multiple times creating 1000 different
pairs of training and testing data before performing cross-validation and
calculating our performance metrics as means for all 1000 different variations.

6. Give at least 2 evaluation metrics and your average performance for each of
   them.  Explain an interpretation of your metrics that says something
   human-understandable about your algorithm’s performance. [relevant rubric
   item: “usage of evaluation metrics”]

Running the final model through the provided tester.py results in the below
output.

```
Pipeline(memory=None,
        steps=[('feature_selection', SelectKBest(k=7, 
                                                 score_func=<function f_classif at 0x107cb4b90>)), 
               ('classification', RandomForestClassifier(bootstrap=True, 
                                                         class_weight='balanced',
                                                         criterion='gini', 
                                                         max_depth=None, 
                                                         max_features='auto', 
                                                         max_leaf_nodes=None, 
                                                         min_impurity_decrease...timators=100, 
                                                         n_jobs=1, 
                                                         oob_score=False, 
                                                         random_state=42, 
                                                         verbose=0, 
                                                         warm_start=False))])

Accuracy:   0.83687   
Precision:  0.40791         Recall: 0.49500 
F1:         0.44726         F2:     0.47473

Total predictions:  15000    
True positives:     990     False positives:    1437    
False negatives:    1010    True negatives:     11563
```

Focusing on the Precision and Recall values we can see that 40% of our model's
predicted POIs are correct (Precision) while 49% of the POIs existing
in the dataset was found successfully (Recall). Lastly, just as a final check
of our reasoning during the feature selection step we try running the model once
more but with the original email features as input instead. 
