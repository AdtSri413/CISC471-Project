import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Import data
    raw_data = pd.read_csv("breast-cancer.csv")

    # Split data into features and target variable
    raw_target = raw_data['diagnosis']
    raw_data = raw_data.drop(columns=['diagnosis'])

    # Find characteristics of each feature (mean, std, etc.)
    stats = raw_data.describe()
    stats = stats.iloc[:, 1:] # Remove ID column
    
    # Plot initial box and whiskers plots
    # for c in stats.columns:
    #     fig1, ax1 = plt.subplots()
    #     ax1.set_title(f'{c} initial box plot')
    #     ax1.boxplot(raw_data[c])
    #     plt.show()

    # Find IQR of features
    iqr = stats.loc['75%'] - stats.loc['25%']
    iqr.name = 'iqr'
    stats = stats.append(iqr)

    # Find whisker endpoints (1.5*IQR from the 25th and 75th percentile)
    min_whisker = stats.loc['25%'] - 1.5*stats.loc['iqr']
    min_whisker.name = 'min_whisker'
    max_whisker = stats.loc['75%'] + 1.5*stats.loc['iqr']
    max_whisker.name = 'max_whisker'
    stats = stats.append(min_whisker)
    stats = stats.append(max_whisker)
    print(stats)

    # Encode target variable (B = 0, M = 1)
    encode_target = raw_target.replace({'B':0,'M':1})

    # Create dataframe to hold all features that we are keeping    
    feature_selected_data = raw_data.iloc[:, 1:].copy()

    # Remove features that are too correlated/not correlated enough with the target variable (pearson correlation)
    for c in feature_selected_data.columns:
        min = stats.loc['min_whisker'][c]
        max = stats.loc['max_whisker'][c]

        selected_attributes = feature_selected_data.loc[feature_selected_data[c] >= min] 
        selected_attributes = selected_attributes.loc[selected_attributes[c] <= max]
        selected_attributes = selected_attributes[c]
        selected_target = encode_target.loc[selected_attributes.index]

        # Calculate pearson correlation coefficient
        corr = np.corrcoef(selected_attributes, selected_target)[0][1]
        
        # Remove features that are too highly or lowly correlated to the target variable
        if abs(corr) > 0.85 or abs(corr) < 0.15:
            feature_selected_data = feature_selected_data.drop(columns=[c])

    # display remaining features (uncomment line below)
    print(feature_selected_data.columns)
