from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    data = data.copy()
    
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: cipher_suite.encrypt(x.encode()).decode())
    
    return data

def decrypt_data(data):
    data = data.copy()
    
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: cipher_suite.decrypt(x.encode()).decode())
    
    return data
