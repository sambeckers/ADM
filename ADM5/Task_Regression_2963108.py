import pandas as pd
import requests
import zipfile
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

# Data loading and preprocessing functions
def load_and_preprocess_data():
    """
    Load and preprocess the bike sharing dataset. This dataset includes dates and other features related to the number of bike rentals.
    Returns the feature matrix X and the target variable y.
    """
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

    # DONE: Extract the 'day.csv' file from the zip archive and load it into the data DataFrame.
    response = requests.get(data_url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('day.csv') as f:
            bike_data = pd.read_csv(f)

    # DONE: Convert the 'dteday' column from a string to a datetime format and extract 
    # and create a 'day_of_month' column from it, 
    # which contains only the "day" information for each date
    bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
    bike_data['day_of_month'] = bike_data['dteday'].dt.day

    #DONE: Remove the original date column and other irrelevant columns ('dteday', 'casual', 'registered', 'cnt').
    # 'dteday' column contains date information and has been specially handled above.
    # 'casual' and 'registered' column would introduce a bias and lead to overfitting
    # 'cnt' column already contains the target information
    # First, set target y from 'cnt'
    y = bike_data['cnt']
    # Then drop target and other irrelevant columns to form feature matrix X
    X = bike_data.drop(columns=['dteday', 'casual', 'registered', 'cnt'])

    return X, y

# Define the function for the random forest regression experiment
def random_forest_regression_experiment(n_estimators, min_samples_leaf, X, y):
    """
    Trains a RandomForestRegressor with the given parameters and performs 10-fold cross-validation.

    Parameters:
    - n_estimators: The number of trees in the forest.
    - min_samples_leaf: The minimum number of samples required for a leaf node.
    - X: Feature matrix (bike features).
    - y: Target vector (number of bike rentals).

    Returns:
    - The average negative mean squared error of the cross-validation.
    """

    # DONE: Create a RandomForestRegressor model with given parameters
    # n_estimators and min_sample_leaf as input, set random_state to 42
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)
    
    # DONE: Implement 10-fold cross-validation with random_state set to 42
    # and compute the mean negative mean squared error using cross-validation
    cross_val = KFold(n_splits=10, random_state=42, shuffle=True) # Must shuffle when using a random state
    mse = cross_val_score(model, X, y, cv=cross_val, scoring='neg_mean_squared_error')

    # Return the computed mean negative mean squared error.
    return mse.mean()

# Function to find the best hyperparameter combination
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    """
    Find the best hyperparameter combination based on cross validation negative mean squared error.

    Parameters:
    - n_estimators_list: list of different values ​​for the number of trees.
    - min_samples_leaf_list: list of different values ​​for the minimum number of leaf samples.
    - X: feature matrix.
    - y: target vector.

    Returns:
    - The best hyperparameter combination and the corresponding negative mean squared error.
    """
    results = []

    # DONE: Iterate over all combinations of hyperparameters, calculate the cross-validation negative mean squared error 
    # using the random_forest_regression_experiment function for each combination, and append the results (n_estimators, 
    # min_samples_leaf, and mse) to the results list.
    for n_est, m_sample in zip(n_estimators_list, min_samples_leaf_list):
        mse = random_forest_regression_experiment(n_est, m_sample, X, y)
        results.append({'n_estimators': n_est, 'min_samples_leaf': m_sample, 'mse': mse})

    # DONE: Select the result with the smallest negative mean squared error (closest to zero)
    # from the results list as the best result.
    best_result = max(results, key=lambda x: x['mse'])

    # Return the best combination of hyperparameters and their corresponding smallest negative mean squared error
    return best_result # best_result is a dictionary

    # DONE: After running the above function, manually input your best result here
def manually_entered_best_params_and_mse():
    best_params = {'n_estimators': 10, 'min_samples_leaf': 1}  # Example, to be replaced with the retrieved best parameters
    best_mse = -459700.5847  # Example mse, to be replaced with your achieved mse
    return best_params, best_mse

if __name__ == "__main__":
    # Experiment with different values for n_estimators and min_samples_leaf
    # to find the best parameters setting among the following options:
    X, y = load_and_preprocess_data()
    n_estimators_list = [10, 50, 100, 1000] # Do not modify these parameter settings, experiment with them
    min_samples_leaf_list = [1, 10, 50, 100] # Do not modify these parameter settings, experiment with them

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y)
    print(f"Best Hyperparameter: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best negative mean square error: {best_params['mse']:.4f}")
