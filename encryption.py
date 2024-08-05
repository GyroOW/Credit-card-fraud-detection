from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    """
    Encrypts the sensitive data in the dataset.
    
    Args:
        data (pandas.DataFrame): The original dataset.
        
    Returns:
        pandas.DataFrame: The encrypted dataset.
    """
    data = data.copy()
    
    # Encrypt the credit card number
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: cipher_suite.encrypt(x.encode()).decode())
    
    return data

def decrypt_data(data):
    """
    Decrypts the sensitive data in the dataset.
    
    Args:
        data (pandas.DataFrame): The encrypted dataset.
        
    Returns:
        pandas.DataFrame: The decrypted dataset.
    """
    data = data.copy()
    
    # Decrypt the credit card number
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: cipher_suite.decrypt(x.encode()).decode())
    
    return data
