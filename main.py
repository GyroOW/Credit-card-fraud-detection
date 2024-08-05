import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import shap
from anonymization import pseudonymize_data, mask_data
from encryption import encrypt_data, decrypt_data
from database import store_flagged_transaction
from data_processing import load_credit_card_data, preprocess_data
from model import train_random_forest_model, tune_model_hyperparameters
from evaluation import evaluate_classification_performance
from interpretability import plot_shap_summary
from real_time_monitoring import monitor_real_time_transactions


def main():
    # Step 1: Load and preprocess the credit card data
    credit_card_data = load_credit_card_data('creditcard.csv')
    X_train, X_test, y_train, y_test = preprocess_data(credit_card_data)
    
    # Step 2: Train a basic random forest model and tune its hyperparameters
    random_forest_model = train_random_forest_model(X_train, y_train)
    tuned_model = tune_model_hyperparameters(random_forest_model, X_train, y_train)
    
    # Step 3: Evaluate the performance of the tuned model on the test data
    evaluate_classification_performance(tuned_model, X_test, y_test)
    
    # Step 4: Use SHAP to understand the model's predictions and visualize feature importance
    plot_shap_summary(tuned_model, X_test)
    
    # Step 5: Set up real-time transaction monitoring to detect potential fraud
    monitor_real_time_transactions(tuned_model)

def load_data(file_path):
    """
    Loads the dataset from the specified file path.
    
    Args:
        file_path (str): The path to the dataset file.
        
    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocesses the data by applying pseudonymization, masking, and encryption.
    
    Args:
        data (pandas.DataFrame): The original dataset.
        
    Returns:
        pandas.DataFrame: The preprocessed dataset.
    """
    data = pseudonymize_data(data)
    data = mask_data(data)
    data = encrypt_data(data)
    return data

def train_model(X_train, y_train):
    """
    Trains a Random Forest model on the training data.
    
    Args:
        X_train (numpy.ndarray): The training features.
        y_train (numpy.ndarray): The training labels.
        
    Returns:
        RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model performance and plots ROC and Precision-Recall curves.
    
    Args:
        model (RandomForestClassifier): The trained model.
        X_test (numpy.ndarray): The testing features.
        y_test (numpy.ndarray): The true labels for the test data.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'ROC-AUC Score: {roc_auc}')
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f'PR-AUC Score: {pr_auc}')
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

def interpret_model(model, X_test):
    """
    Interprets the model using SHAP values and generates plots.
    
    Args:
        model (RandomForestClassifier): The trained model.
        X_test (numpy.ndarray): The testing features.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar')
    plt.show()

    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot("V14", model.predict, X_test)
    plt.show()
    
    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot("V17", model.predict, X_test)
    plt.show()
    
    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot(("V14", "V17"), model.predict, X_test, ice=True, grid_resolution=20)
    plt.show()

def main():
    """
    Main function to load data, preprocess, train model, evaluate, interpret, and store flagged transactions.
    """
    data = load_data('credit_card_transactions.csv')
    data = preprocess_data(data)
    
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    interpret_model(model, X_test)
    
    flagged_indices = data[data['Class'] == 1].index.tolist()
    for idx in flagged_indices:
        store_flagged_transaction(data.loc[idx, 'transaction_id'], 'High fraud probability')

if __name__ == "__main__":
    main()