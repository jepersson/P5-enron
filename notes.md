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
