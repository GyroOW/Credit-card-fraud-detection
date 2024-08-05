import hashlib

def pseudonymize_data(data):
    """
    Pseudonymizes sensitive information in the dataset.
    
    Args:
        data (pandas.DataFrame): The original dataset.
        
    Returns:
        pandas.DataFrame: The pseudonymized dataset.
    """
    data = data.copy()
    
    # Pseudonymize the credit card number
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    
    # Pseudonymize the user ID
    if 'user_id' in data.columns:
        data['user_id'] = data['user_id'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    
    return data

def mask_data(data):
    """
    Masks sensitive parts of the data.
    
    Args:
        data (pandas.DataFrame): The original dataset.
        
    Returns:
        pandas.DataFrame: The masked dataset.
    """
    data = data.copy()
    
    # Mask part of the credit card number
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: x[:4] + '****' + x[-4:])
    
    return data
