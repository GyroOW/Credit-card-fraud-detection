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

if __name__ == '__main__':
    main()
