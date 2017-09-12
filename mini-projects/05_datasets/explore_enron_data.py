#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
import csv

# Start by loading the enron data set.
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl",
                              "r"))

# Check the number of data points (length of the top level dict)
len(enron_data)
# Output: 146

# Check the number of features (length of the second level dict)
len(enron_data.items()[0][1])
# Output: 21

# Check the number of POI entries in the data set
len([x for x in enron_data.items() if x[1]["poi"] == 1])
# Output: 18

# Check the number of persons that are mentioned as POIs in the provided CSV
with open("../final_project/poi_names.txt", "rb") as f:
    r = csv.reader(f, delimiter=" ")
    # Skip the first two lines in the CSV that doesn't contain any data entries
    r.next()
    r.next()
    # Read in the rest of the file as row by row into a list
    pois = [row for row in r]
# Count the length to get the number of POIs
len(pois)
# Output: 35

# What is the total value of the stock beloning to James Prentice?
enron_data["PRENTICE JAMES"]["total_stock_value"]
# Output: 1095040

# How many messages do we have from Wesley Colwell to POIs?
enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
# Output: 11

# What's the value of stock options exercised by Jeff Skilling?
enron_data["SKILLING JEFF"]
# Output: "KeyError: 'SKILLING JEFF'"

# Can't seem to find a Jeff Skilling among our keys so instead look for any
# Skilling
[key for key, value in enron_data.items() if "skilling" in key.lower()]
# Output: ['SKILLING JEFFREY K']

# What's the value of stock options exercised by Jeffrey K Skilling?
enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
# Output: 19250000

# Out of the top three individuals at enron (Lay, Skilling, Fastow), who took
# the most money("total_payments")?
# Start with finding the proper names for Lay and Fastow.
[key for key, value in enron_data.items() if "lay" in key.lower()]
[key for key, value in enron_data.items() if "fastow" in key.lower()]
# Output: ['LAY KENNETH L'] and ['FASTOW ANDREW S']

# Then find their total_payments
top_three = ["LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"]
[{key: value["total_payments"]}
 for key, value in enron_data.items() if key in top_three]
# Output: [{'LAY KENNETH L': 103559793},
#          {'FASTOW ANDREW S': 2424083},
#          {'SKILLING JEFFREY K': 8682716}]

# How many folks in this data set have a quantified salary?
len([key
     for key, value in enron_data.items() if value["salary"] != "NaN"])
# Output: 95
