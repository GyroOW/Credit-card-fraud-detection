from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest_model(X_train, y_train):
    """
    Train a Random Forest classifier on the provided training data.

    Args:
    - X_train (numpy.ndarray): Training features.
    - y_train (numpy.ndarray): Training labels.

    Returns:
    - RandomForestClassifier: The trained Random Forest model.
    """
    # Create and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_model_hyperparameters(model, X_train, y_train):
    """
    Tune the hyperparameters of the model using GridSearchCV to find the best parameters.

    Args:
    - model (RandomForestClassifier): The base model to be tuned.
    - X_train (numpy.ndarray): Training features.
    - y_train (numpy.ndarray): Training labels.

    Returns:
    - RandomForestClassifier: The best model found by GridSearchCV.
    """
    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform GridSearchCV to find the best parameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_
