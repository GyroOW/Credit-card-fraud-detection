import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_credit_card_data(file_path):
    """
    Load the credit card transaction data from a CSV file.

    Args:
    - file_path (str): The path to the CSV file containing the data.

    Returns:
    - pandas.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the credit card transaction data for modeling.
    
    Here's what this function does:
    1. Splits the data into features (X) and labels (y).
    2. Splits the data into training and testing sets.
    3. Standardizes the features to have zero mean and unit variance.
    
    Args:
    - data (pandas.DataFrame): The raw dataset.

    Returns:
    - tuple: Contains the preprocessed training and testing data.
      - X_train (numpy.ndarray): Training features.
      - X_test (numpy.ndarray): Testing features.
      - y_train (numpy.ndarray): Training labels.
      - y_test (numpy.ndarray): Testing labels.
    """
    # Step 1: Separate features and labels
    features = data.drop('Class', axis=1)
    labels = data['Class']

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
