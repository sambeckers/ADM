# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from ucimlrepo import fetch_ucirepo 

# Import data From UCI repository about Portuguese wine qualities:
# This dataset contains features related to wine properties and the labels 
# related to the different wine qualities. Our goal is to predict the correct
# wine quality just looking at the features and not at the labels. 
wine_quality = fetch_ucirepo(id=186) 
  
X = wine_quality.data.features.to_numpy()
y = wine_quality.data.targets.to_numpy().ravel() # sklearn expects shape (n_samples,)

# Define the function for Random Forest classification with cross-validation
def random_forest_experiment(n_estimators, min_samples_leaf, X, y):
    """
    Trains a RandomForestClassifier with specified parameters and performs 10-fold cross-validation.
    
    Parameters:
    - n_estimators: Number of trees in the forest.
    - min_samples_leaf: Minimum number of samples required to be at a leaf node.
    - X: Feature matrix (wine features).
    - y: Target vector (wine quality labels).
    
    Returns:
    - Mean cross-validation accuracy (measure to estimate the accuracy of our prediction
      based on a 10-fold cross validation)
    """
    # DONE: Create a RandomForestClassifier model with given parameters
    # n_estimators and min_sample_leaf as input, set random_state to 2024
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=2024)

    # DONE: Implement 10-fold cross-validation with random_state set to 2024
    # and compute the scores based on cross-validation accuracy
    cross_val = KFold(n_splits=10, random_state=2024, shuffle=True) # Must shuffle when using a random state
    scores = cross_val_score(model, X, y, cv=cross_val, scoring='accuracy')

    # Return the average cross-validation accuracy
    return scores.mean()

# Function to return the best result based on hyperparameters
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    """
    Finds the best hyperparameter combination based on cross-validation accuracy.
    
    Parameters:
    - n_estimators_list: List of different values for the number of trees.
    - min_samples_leaf_list: List of different values for min_samples_leaf.
    - X: Feature matrix (wine features).
    - y: Target vector (wine quality labels).
    
    Returns:
    - Best hyperparameter combination and corresponding accuracy.
    """
    results = []

    # DONE: Iterate over all combinations of hyperparameters and calculate the 
    # cross-validation accuracy by using random_forest_experiment function, append 
    # n_estimators, min_sample_leaf and accuracy to results.
    for n_est, m_sample in zip(n_estimators_list, min_samples_leaf_list):
        accuracy = random_forest_experiment(n_est, m_sample, X, y)
        results.append({'n_estimators': n_est, 'min_samples_leaf': m_sample, 'accuracy': accuracy})
    
    # Return the best combination of hyperparameters and the corresponding accuracy
    best_result = max(results, key=lambda x: x['accuracy'])
    return best_result

    # DONE: After running the above function, manually input your best result here
def manually_entered_best_params_and_accuracy():
    best_params = {'n_estimators': 10, 'min_samples_leaf': 1}  # Example, to be replaced with the retrieved best parameters
    best_accuracy = 0.6749  # Example accuracy, to be replaced with your achieved accuracy
    return best_params, best_accuracy
    
# Main function
if __name__ == "__main__":
    # Experiment with different values for n_estimators and min_samples_leaf
    # to find the best parameters setting among the following options:
    n_estimators_list = [10, 50, 100, 1000]
    min_samples_leaf_list = [1, 10, 50, 100]

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y)

    print(f"Best Hyperparameters: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best Accuracy: {best_params['accuracy']:.4f}")
