from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model_with_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    
    return best_rf
