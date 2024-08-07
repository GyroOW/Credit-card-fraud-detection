import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import time

from data_validation import load_data
from pipeline import build_pipeline
from model_training import train_model_with_tuning
from model_evaluation import evaluate_model
from model_interpretation import interpret_model
from database import store_flagged_transaction
from profiling import profile

def main():
    start_time = time.time()
    try:
        data = load_data('credit_card_transactions.csv')
        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        data_pd = data.toPandas()
        X = data_pd.drop(columns=['Class'])
        y = data_pd['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Data split in {time.time() - start_time:.2f} seconds.")
        
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        start_time = time.time()
        model = train_model_with_tuning(X_train, y_train)
        logging.info(f"Model trained in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        pipeline = build_pipeline(model, numeric_features)
        pipeline.fit(X_train, y_train)
        logging.info(f"Pipeline fitted in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        evaluate_model(pipeline, X_test, y_test)
        logging.info(f"Model evaluated in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        interpret_model(pipeline.named_steps['model'], X_test)
        logging.info(f"Model interpretation completed in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        flagged_indices = data_pd[data_pd['Class'] == 1].index.tolist()
        for idx in flagged_indices:
            store_flagged_transaction(data_pd.loc[idx, 'transaction_id'], 'High fraud probability')
        logging.info(f"Flagged transactions stored in {time.time() - start_time:.2f} seconds.")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    profile(main)
