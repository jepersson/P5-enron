#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    # your code goes here

    # Calculate the residual  errors
    residual_errors = predictions - net_worths

    # Create the right data structure with all data still contained
    uncleaned_data = zip(ages, net_worths, residual_errors)

    # Sort the newly built list based on residual_errors and put the first 81
    # entries in cleaned data
    cleaned_data = sorted(uncleaned_data, key=lambda item: item[2])[:80]

    return cleaned_data
