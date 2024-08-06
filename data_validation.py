import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    data.fillna(0, inplace=True)
    
    if 'credit_card_number' in data.columns:
        data['credit_card_number'] = pd.to_numeric(data['credit_card_number'], errors='coerce')
    
    return data
