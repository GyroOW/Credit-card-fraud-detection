from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification_performance(model, X_test, y_test):
    """
    Evaluate the performance of the classification model on the test data.

    Args:
    - model (RandomForestClassifier): The trained model.
    - X_test (numpy.ndarray): Testing features.
    - y_test (numpy.ndarray): True labels for the test data.

    Prints:
    - Evaluation metrics: Accuracy, Precision, Recall, F1 Score.
    """
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
