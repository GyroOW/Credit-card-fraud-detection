import hashlib

def pseudonymize_data(data):
    data = data.copy()
    
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    
    if 'user_id' in data.columns:
        data['user_id'] = data['user_id'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    
    return data

def mask_data(data):
    data = data.copy()
    
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = data['credit_card_number'].apply(lambda x: x[:4] + '****' + x[-4:])
    
    return data
