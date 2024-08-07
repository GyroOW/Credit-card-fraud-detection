from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def train_model_with_tuning(X_train, y_train):
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 20)
    }
    
    rf = RandomForestClassifier(random_state=42)
    randomized_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, scoring='roc_auc', random_state=42)
    randomized_search.fit(X_train, y_train)
    
    best_rf = randomized_search.best_estimator_
    
    return best_rf
