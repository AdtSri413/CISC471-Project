''' This file contains various data exploration procedures
'''
import csv
from statistics import mean, median
import numpy as np
import matplotlib.pyplot as plt


''' pretty printer for dicts
    Parameters: 
        d: A dict (dict)
    Output: None
'''
def pprint(d):
    pprint_help(d)
    print()


''' pretty printer for dicts helper function
    Parameters: 
        d: A dict (dict)
        tab: Number of tabs to print at the beginning of a line (int)
        long_feat_len: Length of the longest feature (int)
    Output: None
'''
def pprint_help(d, tab=0, long_feat_len=0):
    if type(d) is dict:
        # Get keys of dict
        features = list(d.keys())

        # Find the length of the longest feature name
        for i in features:
            if len(i)>long_feat_len:
                long_feat_len = len(i)

        # print dict
        for i in features:
            if i is not None:
                if type(d[i]) is dict or type(d[i]) is list:
                    print()
                print(f"{tab*' '}{i}:{(long_feat_len-len(i)+1)*' '}", end="")
                if type(d[i]) is dict:
                    print()
                if type(d[i]) is list:
                    pprint_help(d[i], tab+4, long_feat_len)
                else:
                    pprint_help(d[i], tab+4, tab)

    elif type(d) is list:
        print(d[0])
        for i in d[1:]:
            print(f"{(tab-4)*' '}{(long_feat_len+2)*' '}{i}")

    else:
        # print
        print(d)



''' Reading and formatting dataset
    Parameters: 
        file: data file (string)
    Output: 
        full_dataset_dict: The processed data (dict)
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
    for row in full_dataset:
        for attribute in range(0, len(row)):
            try:
                full_dataset_dict[features[attribute]].append(float(row[attribute]))
            except ValueError:
                full_dataset_dict[features[attribute]].append(row[attribute])
    
    return full_dataset_dict


''' Evaluation of mean and variance of a feature on full dataset
    Parameters: 
        data: The imported data (dict)
        feature: The feature who's mean and variance will be calculated (string)
    Output: 
        dict conatining mean and variance (dict)
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
    Parameters: 
        data: the imported data (dict)
        feature: The feature who's IQR needs to be calculated (string)
    Output: 
        iqr: The IQR for the provided data (float)
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
    Parameters: 
        data: The imported data (dict)
        feature: Feature who's whiskers need to be found (string)
        type: Specifies if the min whisker, max whisker, or both whiskers should be calculated (string)
    Output: 
        Either the value of the smallest whisker, largest whisker, or both (dict)
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
    Parameters: 
        data: The imported data (dict)
        feature: The feature who's outliers need to be calculated (string)
    Output: 
        outliers: A dict of outliers. The key corresponds to the attribute/sample 
            number and the value is the value of the outlier (dict)
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
            outliers[f"{i}"] = values[i]

    return outliers


''' Normalizes data such that it falls between 0 and 1
    Parameters: 
        data: All the imported data (dict)
        feature: The feature to normalize (string)
    Output: None
'''
def normalization(data, feature):
    normalized = []
    minimum = min(data[feature])
    maximum = max(data[feature])

    for i in data[feature]:
        temp = (i - minimum) / (maximum - minimum)
        normalized.append(temp)

    return normalized


''' Remove outliers from a dataset
'''
def remove_outliers_in_dataset(data):
    # Initialize variables
    dataset_no_outliers = {}
    features = list(data.keys())
    outlier_attributes = []

    # Get all indices of outliers
    for i in features:
        o = list(iqr_outliers(data, i).keys())
        for j in o:
            outlier_attributes.append(int(j))

    # Remove duplicates
    outlier_attributes.sort()
    outlier_attributes = list(dict.fromkeys(outlier_attributes))
    
    # Initialize dict keys
    for i in features:
        dataset_no_outliers[i] = []

    # Remove outliers
    for i in range(0, len(data[features[0]])):
        if i not in outlier_attributes:
            for j in features:
                dataset_no_outliers[j].append(data[j][i])

    return dataset_no_outliers


''' Calculates the Pearson correlation coefficient for 2 features. 
    Can be used for 2 numerical features.
    Parameters:
        x: feature1 attributes (list)
        y: feature2 attributes (list)
        feature1: first feature (string)
        feature2: second feature (string)
    Output:
        Pearson correlation coefficient with 4 decimal places (float)
'''
def pearson_corr_coeff(x, y, feature1, feature2):
    # Remove outliers
    no_outliers = remove_outliers_in_dataset({feature1:x, feature2:y})

    x_simple = np.array(no_outliers[feature1])
    y_simple = np.array(no_outliers[feature2])
    my_rho = np.corrcoef(x_simple, y_simple)
    return round(my_rho[0][1], 4)


''' Prints information about the specified feature
    Parameters: 
        data: All the imported data (dict)
        feature: The feature who's info you want to print (string)
    Output: None
'''
def print_info(data, feature):
    # Print attributes of the box plot
    print(f"\n------------{feature.upper()}------------\n")
    
    mean_var = mean_variance(data, feature)

    print(f"Min val     = {min(data[feature])}")
    print(f"Max val     = {max(data[feature])}")
    print(f"IQR         = {iqr(data, feature)}")
    print(f"Median      = {median(data[feature])}")
    print(f"Mean        = {mean_var['mean']}")
    print(f"Variance    = {mean_var['variance']}")
    print(f"Min whisker = {whisker_values(data, feature, 'min')['min']}")
    print(f"Max whisker = {whisker_values(data, feature, 'max')['max']}")
    print("Outliers: ")
    outliers = iqr_outliers(data, feature)
    
    # define variables for length of the longest feature name
    long_feat_len = 0

    # Find the length of the longest feature name
    for i in outliers:
        if len(i)>long_feat_len:
            long_feat_len = len(i)

    for i in outliers:
        print(f"\tAttribute # {i}:{(long_feat_len-len(i)+1)*' '}{outliers[i]}")

    print(f"\n------------{len(feature)*'-'}------------\n")


''' Create box and whiskers plot for the designated feature
    Parameters:
        data: The data associated with the required feature (list)
        feature: The name of the feature (string)
    Output: None
'''
def plot_box(data, feature):
    fig1, ax1 = plt.subplots()
    ax1.set_title(f'{feature} box plot')
    ax1.boxplot(data)
    plt.show()


''' Create scatter plot for 2 features (transformed or not)
    Parameters:
        data1: The data for the first feature (list)
        data2: The data for the second feature (list)
        feature1: The name of the first feature (string)
        feature2: The name of the second feature (string)
    Output: None
'''
def plot_scatter(data1, data2, feature1, feature2):
    fig1, ax1 = plt.subplots()
    ax1.set_title(f"Plot {feature1} vs. {feature2}")
    ax1.scatter(data1, data2)
    plt.show()


''' Main function
'''
if __name__ == "__main__":  
    # import data
    raw_data = import_data("breast-cancer.csv")

    # data that will be cleaned
    data = raw_data 

    # Get list of features
    features = list(data.keys())

    # Remove ID column. It is irrelevent
    del data[features[0]]
    # Update features
    features = features[1:]

    # Replace B and M in target feature attributes with 0 and 1 (0 = B, 1 = M)
    numerical_diagnosis = []
    for i in data[features[0]]:
        if i == 'B':
            numerical_diagnosis.append(0)
        else:
            numerical_diagnosis.append(1)
    data[features[0]] = numerical_diagnosis

    # initialize dictionary to store mean and variance
    mean_var_dict = {}

    # calculate mean and variance for each numerical feature
    for i in features[1:]:
        mean_var_dict[i] = mean_variance(data, i)    

    # Print mean and variance of each numerical feature
    # pprint(mean_var_dict)

    # Plot stuff
    # plot_box(data["radius_mean"], "radius_mean")
    # plot_scatter([number ** 1.4 for number in normalization(data, features[2])], normalization(data, features[5]), features[2], features[5])

    # data1 = "radius_mean"
    # data2 = "diagnosis"
    # plot_scatter(data[data1], data[data2], data1, data2)
    # plot_scatter(normalization(data, data1), data[data2], data1, data2)


    # Remove features that are too correlated/not correlated enough to the target variable
    for i in features[1:]:
        corr_coeff = pearson_corr_coeff(data[features[0]], data[i], features[0], i)
        if corr_coeff<-0.9 or (corr_coeff>-0.1 and corr_coeff<0.1) or corr_coeff>0.9:
            del data[i]
            features.remove(i)
        # print(f"Pearson correlation coefficient {i}{(len('fractal_dimension_worst')-len(i))*' '} = {corr_coeff}")

    # Remove features too highly correlated with other features:
    corr_coeff = {}
    for i in range(0, len(features)):
        corr_coeff[features[i]] = {}
        for j in range(len(features[:i+1]), len(features)):
            corr_coeff[features[i]][features[j]] = pearson_corr_coeff(data[features[i]], data[features[j]], features[i], features[j])
            # print(f"Correlation between {features[i]} {features[j]}       = {corr_coeff[features[i]][features[j]]}")

    pprint(corr_coeff)

    # plot highly correlated features
    for i in range(0, len(features)):
        for j in range(len(features[:i+1]), len(features)):
            if corr_coeff[features[i]][features[j]]<-0.9 or corr_coeff[features[i]][features[j]]>0.9:
                no_outliers = remove_outliers_in_dataset({features[i]: data[features[i]], features[j]: data[features[j]]})
                plot_scatter(normalization(no_outliers, features[i]), normalization(no_outliers, features[j]), features[i], features[j])

    # print_info(data, features[1])