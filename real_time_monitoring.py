from kafka import KafkaConsumer
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

def monitor_real_time_transactions(model):
    """
    Set up real-time monitoring for credit card transactions using Kafka and detect potential fraud.

    Args:
    - model (RandomForestClassifier): The trained model for predicting fraud probability.
    """
    # Connect to the Kafka consumer
    consumer = KafkaConsumer('credit_card_transactions',
                             bootstrap_servers=['localhost:9092'],
                             group_id='fraud_detection_group')
    
    scaler = StandardScaler()  # Assuming you have initialized the scaler in data_processing.py
    
    for message in consumer:
        # Decode the transaction data from Kafka
        transaction_data = json.loads(message.value.decode('utf-8'))
        transaction_df = pd.DataFrame(transaction_data, index=[0])
        transaction_scaled = scaler.transform(transaction_df)
        
        # Predict the fraud probability for the transaction
        fraud_probability = model.predict_proba(transaction_scaled)[0, 1]
        
        # Print an alert if the fraud probability is above 50%
        if fraud_probability > 0.5:
            print(f"Potential fraud detected with probability: {fraud_probability}")
        else:
            print("Transaction is likely not fraudulent.")
