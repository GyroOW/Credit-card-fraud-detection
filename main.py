import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from data_validation.py import load_data
from pipeline import build_pipeline
from model_training import train_model_with_tuning
from model_evaluation import evaluate_model
from model_interpretation import interpret_model
from database import store_flagged_transaction

def main():
    try:
        data = load_data('credit_card_transactions.csv')
        
        X = data.drop(columns=['Class'])
        y = data['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = train_model_with_tuning(X_train, y_train)
        pipeline = build_pipeline(model)
        
        pipeline.fit(X_train, y_train)
        
        evaluate_model(pipeline, X_test, y_test)
        
        interpret_model(pipeline.named_steps['model'], X_test)
        
        flagged_indices = data[data['Class'] == 1].index.tolist()
        for idx in flagged_indices:
            store_flagged_transaction(data.loc[idx, 'transaction_id'], 'High fraud probability')
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
