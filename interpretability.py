import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, X_test):
    """
    Create and display a SHAP summary plot to show feature importance.

    Args:
    - model (RandomForestClassifier): The trained model.
    - X_test (numpy.ndarray): Testing features.
    """
    # Initialize the SHAP explainer and compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Plot the SHAP summary plot
    shap.summary_plot(shap_values, X_test, plot_type='bar')
