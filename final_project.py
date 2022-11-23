import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# calculate and print the confusion matrix, accuracy, sensitivity, and specificity of given predicted and actual labels
def print_stats(testing_actual, testing_pred, name):
    print(pd.crosstab(pd.Series(testing_actual, name='Actual'), pd.Series(testing_pred, name='Predicted')), "\n")

    accuracy = accuracy_score(testing_actual, testing_pred)
    print(f"accuracy of {name} model= %.3f" % accuracy)

    sensitivity = recall_score(testing_actual, testing_pred)
    print(f"sensitivity of {name} model= %.3f" % sensitivity)
    
    tn, fp, fn, tp = confusion_matrix(testing_actual, testing_pred).ravel()
    specificity = tn/(tn+fp)
    print(f"specificity of {name} model= %.3f" % specificity)


if __name__ == "__main__":
    # Import data
    raw_data = pd.read_csv("breast-cancer.csv")

    # Split data into features and target variable
    raw_target = raw_data['diagnosis']
    raw_data = raw_data.drop(columns=['diagnosis'])

    # Find characteristics of each feature (mean, std, etc.)
    stats = raw_data.describe()
    stats = stats.iloc[:, 1:] # Remove ID column
    
    # Plot initial box and whiskers plots (uncomment block below)
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
    # print(stats)

    # Encode target variable (B = 0, M = 1)
    encode_target = raw_target.replace({'B':0,'M':1})

    # Create dataframe to hold all features that we are keeping    
    feature_selected_data = raw_data.iloc[:, 1:].copy()

    # Drop the "worst" category (we will not be analysing it)
    feature_selected_data = feature_selected_data.drop([x for x in feature_selected_data.columns if "worst" in x], axis=1)

    # Remove features that are too correlated/not correlated enough with the target variable (pearson correlation)
    for c in feature_selected_data.columns:
        min = stats.loc['min_whisker'][c]
        max = stats.loc['max_whisker'][c]

        # Remove outliers (based on 1.5*IQR from 25th and 75th percentile)
        selected_attributes = feature_selected_data.loc[feature_selected_data[c] >= min] # Remove samples below the lower whisker
        selected_attributes = selected_attributes.loc[selected_attributes[c] <= max] # Remove samples above the upper whisker
        selected_attributes = selected_attributes[c] # Select feature c
        selected_target = encode_target.loc[selected_attributes.index] # select corresponding samples from target variable

        # Calculate pearson correlation coefficient
        corr = np.corrcoef(selected_attributes, selected_target)[0][1]
        
        # Remove features that are too highly or lowly correlated to the target variable
        if abs(corr) > 0.85 or abs(corr) < 0.15:
            feature_selected_data = feature_selected_data.drop(columns=[c])
            stats = stats.drop(columns=c)

    # Split data into two categories: mean and se
    mean_data = feature_selected_data[[x for x in feature_selected_data.columns if "mean" in x]]
    se_data = feature_selected_data[[x for x in feature_selected_data.columns if "se" in x]]

    # Remove features too highly correlated with other features
    corr_coeff = []
    for data in [mean_data, se_data]: # Investigate each group of data separately
        for i in range(0, len(data.columns)):
            for j in range(i+1, len(data.columns)):
                # Find min and max whiskers
                min_i = stats.loc['min_whisker'][data.columns[i]]
                max_i = stats.loc['max_whisker'][data.columns[i]]
                min_j = stats.loc['min_whisker'][data.columns[j]]
                max_j = stats.loc['max_whisker'][data.columns[j]]

                # Remove outliers
                selected_attributes = data.loc[data.iloc[:, i] >= min_i]
                selected_attributes = selected_attributes.loc[selected_attributes.iloc[:, i] <= max_i]
                selected_attributes = selected_attributes.loc[selected_attributes.iloc[:, j] >= min_j]
                selected_attributes = selected_attributes.loc[selected_attributes.iloc[:, j] <= max_j]

                # Find pearson correlation coefficient. If it is too high, add the feature pair to a list for further analysis
                corr = np.corrcoef(selected_attributes[data.columns[i]], selected_attributes[data.columns[j]])[0][1]
                if corr > 0.85:
                    corr_coeff.append([data.columns[i], data.columns[j]])

    # Print pairs of features that are highly correlated (uncomment line below)
    # print(corr_coeff)

    # After investigating the highly correlated feature pairs, we have decided to remove the following features:
    mean_data = mean_data.drop(columns=["perimeter_mean", "area_mean", "concavity_mean"])
    se_data = se_data.drop(columns=["perimeter_se", "area_se"])

    # Remove outliers from final selected features
    for c in mean_data.columns:
        min = stats.loc['min_whisker'][c]
        max = stats.loc['max_whisker'][c]

        # Remove outliers
        mean_data = mean_data.loc[mean_data[c] >= min]
        mean_data = mean_data.loc[mean_data[c] <= max]

    for c in se_data.columns:
        min = stats.loc['min_whisker'][c]
        max = stats.loc['max_whisker'][c]

        # Remove outliers
        se_data = se_data.loc[se_data[c] >= min]
        se_data = se_data.loc[se_data[c] <= max]

    # Get stats for each data set
    mean_stats = mean_data.describe()
    se_stats = se_data.describe()

    # Get target variable for remaining samples in each data set
    mean_target = encode_target.loc[mean_data.index]
    se_target = encode_target.loc[se_data.index]

    # Normalize the data
    mean_data_norm = normalize(mean_data)
    mean_data_norm = pd.DataFrame(mean_data_norm, columns=mean_data.columns)
    se_data_norm = normalize(se_data)
    se_data_norm = pd.DataFrame(se_data_norm, columns=se_data.columns)
    


    
    # DECISION TREE MODEL

    # Use GridSearchCV from sklearn to try to determine the best hyperparameters
    decisionTreeModel = GridSearchCV(DecisionTreeClassifier(), 
        param_grid={'criterion':['gini','entropy'],'max_depth':[4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}, 
        cv=5, 
        scoring="accuracy")

    # MEAN

    # Split dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(mean_data, mean_target, test_size=0.2)

    # Fit the model to the training data
    decisionTreeModel.fit(X_train, y_train)
    best_decisionTreeModel = decisionTreeModel.best_estimator_

    # Predict the target values (AML/0 or ALL/1) of the testing data
    testing_pred = best_decisionTreeModel.predict(X_test)

    # Calculate accuracy, sensitivity (recall), specificity, and confusion matrix
    print("\n----------------DECISION TREE MODEL - MEAN----------------\n")
    print("Tuned Hyperparameters :", decisionTreeModel.best_params_)
    print("Accuracy :\n",decisionTreeModel.best_score_)
    print_stats(np.array(y_test), testing_pred, "decision tree")

    # SE

    # Split dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(se_data, se_target, test_size=0.2)

    # Fit the model to the training data
    decisionTreeModel.fit(X_train, y_train)
    best_decisionTreeModel = decisionTreeModel.best_estimator_

    # Predict the target values (AML/0 or ALL/1) of the testing data
    testing_pred = best_decisionTreeModel.predict(X_test)

    # Calculate accuracy, sensitivity (recall), specificity, and confusion matrix
    print("\n----------------DECISION TREE MODEL - SE----------------\n")
    print("Tuned Hyperparameters :", decisionTreeModel.best_params_)
    print("Accuracy :\n",decisionTreeModel.best_score_)
    print_stats(np.array(y_test), testing_pred, "decision tree")




    # LOGISTIC REGRESSION MODEL

    # Use GridSearchCV from sklearn to try to determine the best hyperparameters
    logisticRegModel = GridSearchCV(LogisticRegression(solver="liblinear"), 
        param_grid={'penalty': ['l1', 'l2'], 'C': np.logspace(-3,3,7)}, 
        cv=3, 
        scoring="accuracy")

    # MEAN

    # Split dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(mean_data, mean_target, test_size=0.2)

    # Fit the model to the training data
    logisticRegModel.fit(X_train, y_train)
    best_logisticRegModel = logisticRegModel.best_estimator_

    # Predict the target values (AML/0 or ALL/1) of the testing data
    testing_pred = best_logisticRegModel.predict(X_test)

    # Calculate accuracy, sensitivity (recall), specificity, and confusion matrix
    print("\n----------------LOGISTIC REGRESSION MODEL - MEAN----------------\n")
    print("Tuned Hyperparameters :", logisticRegModel.best_params_)
    print("Accuracy :\n",logisticRegModel.best_score_)
    print_stats(np.array(y_test), testing_pred, "logistic regression")

    # SE

    # Split dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(se_data, se_target, test_size=0.2)

    # Fit the model to the training data
    logisticRegModel.fit(X_train, y_train)
    best_logisticRegModel = logisticRegModel.best_estimator_

    # Predict the target values (AML/0 or ALL/1) of the testing data
    testing_pred = best_logisticRegModel.predict(X_test)

    # Calculate accuracy, sensitivity (recall), specificity, and confusion matrix
    print("\n----------------LOGISTIC REGRESSION MODEL - SE----------------\n")
    print("Tuned Hyperparameters :", logisticRegModel.best_params_)
    print("Accuracy :\n",logisticRegModel.best_score_)
    print_stats(np.array(y_test), testing_pred, "logistic regression")
