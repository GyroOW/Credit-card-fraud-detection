from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from anonymization import pseudonymize_data, mask_data
from encryption import encrypt_data

def build_pipeline(model, numeric_features):
    preprocess = ColumnTransformer(
        transformers=[
            ('pseudonymize', FunctionTransformer(pseudonymize_data), ['credit_card_number', 'user_id']),
            ('mask', FunctionTransformer(mask_data), ['credit_card_number']),
            ('encrypt', FunctionTransformer(encrypt_data), ['credit_card_number']),
            ('scale', StandardScaler(), numeric_features)
        ]
    )
    
    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('model', model)
    ])
    
    return pipeline
