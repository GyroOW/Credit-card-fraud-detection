from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    """
    Detect anomalies in the data using Isolation Forest.

    Args:
    - data (pandas.DataFrame): The data to be analyzed.

    Returns:
    - list: A list of indices flagged as anomalies.
    """
    clf = IsolationForest(contamination=0.1, random_state=42)
    preds = clf.fit_predict(data)
    anomaly_indices = data[preds == -1].index.tolist()
    
    return anomaly_indices
