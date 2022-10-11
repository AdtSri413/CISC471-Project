''' This file contains various data exploration procedures
'''
import csv
from statistics import mean, median
from xml.dom import minicompat
import numpy as np
import matplotlib.pyplot as plt

'''Reading and formatting dataset
   Parameters: data file (string)
   Output: Data (dict)
'''
def import_data(file):
    # Read full csv dataset file
    full_dataset_file = open(file)
    full_dataset = csv.reader(full_dataset_file)

    # Extract the feature names
    features = []
    features = next(full_dataset)

    # Initialize dictionary to store data
    full_dataset_dict = {}

    # Add features to dictionary as keys
    for i in features:
        full_dataset_dict[i] = []

    # Add all attributes as values for their respective features. The dictionary values are stored as lists
    rows = []
    for row in full_dataset:
        for attribute in range(0, len(row)):
            try:
                full_dataset_dict[features[attribute]].append(float(row[attribute]))
            except ValueError:
                full_dataset_dict[features[attribute]].append(row[attribute])
    
    return full_dataset_dict


''' Evaluation of mean and variance of a feature on full dataset
    Parameters: data (dict), feature (string)
    Output: mean, variance (dict)
'''
def mean_variance(data, feature):
    # initialize list containing values in the specified feature
    values = []
    # getting values
    try:
        values = data[feature]
    except ValueError:
        print("Error: Unknown feature")
        return
    
    # calculating mean and variance
    m = mean(values)
    variance = np.var(values)

    # Return mean and variance
    return {"mean": m, "variance": variance}


''' Get iqr of a feature
    Parameters: data (dict), feature (string)
    Output: The IQR for the provided data (float)
'''
def iqr(data, feature):
    values = []
    # getting values
    try:
        values = data[feature]
    except ValueError:
        print("Error: Unknown feature")
        return
    
    iqr = np.percentile(values, 75) - np.percentile(values, 25)

    return iqr


''' Get minimum/maximum box plot whisker value
    Parameters: data (dict), feature (string), type (string) (either min, max, or both)
    Output: Either the value of the smallest whisker, largest whisker, or both (dict)
'''
def whisker_values(data, feature, type="both"):
    values = []
    # getting values
    try:
        values = data[feature]
    except ValueError:
        print("Error: Unknown feature")
        return

    # Make sure type is valid
    if type != "both" and type != "min" and type != "max":
        print("Error: Invalid type parameter")
        return
    
    ret = {}

    # If type is min or both, find minimum whisker value
    if type == "min" or type == "both":
        ret["min"] = np.percentile(values, 25) - 1.5*iqr(data, feature)
    if type == "max" or type == "both":
        ret["max"] = np.percentile(values, 75) + 1.5*iqr(data, feature)

    return ret


''' Get outliers based on IQR
    Parameters: data (dict), feature (string)
    Output: outliers and their index (dict)
'''
def iqr_outliers(data, feature):
    values = []
    # getting values
    try:
        values = data[feature]
    except ValueError:
        print("Error: Unknown feature")
        return
    
    # Get min and max whisker values
    min_whisker = whisker_values(data, feature, "min")["min"]
    max_whisker = whisker_values(data, feature, "max")["max"]

    # initialize dict of outliers
    outliers = {}

    # for each attribute, if the attribute falls outside the range of min_whisker to max_whisker, add it to the outlier dict
    for i in range(0, len(values)):
        if values[i] < min_whisker or values[i] > max_whisker:
            outliers[f"attribute {i}"] = values[i]

    return outliers


''' pretty printer for dicts
    Parameters: A dict (dict)
    Output: None
    This works best with non-nested dicts
'''
def pprint(dict):
    # Get keys of dict
    features = list(dict.keys())

    # define variables for length of the longest feature name
    long_feat_len = 0

    # Find the length of the longest feature name
    for i in features:
        if len(i)>long_feat_len:
            long_feat_len = len(i)

    # print dict
    for i in features:
        print(f"{i}:{(long_feat_len-len(i)+1)*' '}{dict[i]}")


''' Main function
'''
if __name__ == "__main__":  
    # import data
    data = import_data("breast-cancer.csv")

    # initialize dictionary to store mean and variance
    mean_var_dict = {}

    # Get list of features
    features = list(data.keys())

    # calculate mean and variance for each numerical feature
    for i in features[2:]:
        mean_var_dict[i] = mean_variance(data, i)    

    # Print mean and variance of each numerical feature
    pprint(mean_var_dict)


    # Ex. Create box plot for feature #2: radius_mean
    fig1, ax1 = plt.subplots()
    ax1.set_title(f'{features[2]} box plot')
    ax1.boxplot(data[features[2]])
    d = plt.boxplot(data[features[2]])

    # Print attributes of the box plot
    print(f"Min val     = {min(data[features[2]])}")
    print(f"Max val     = {max(data[features[2]])}")
    print(f"IQR         = {iqr(data, features[2])}")
    print(f"Median      = {median(data[features[2]])}")
    print(f"Min whisker = {whisker_values(data, features[2], 'min')['min']}")
    print(f"Max whisker = {whisker_values(data, features[2], 'max')['max']}")
    print("Outliers: ")
    pprint(iqr_outliers(data, features[2]))

    plt.show()
