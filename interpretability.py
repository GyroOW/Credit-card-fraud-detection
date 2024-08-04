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
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar')
    plt.show()
    
    # Plot partial dependence plots (PDP) for selected features
    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot("V14", model.predict, X_test)
    plt.show()
    
    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot("V17", model.predict, X_test)
    plt.show()
    
    # Plot individual conditional expectation (ICE) plots
    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot(("V14", "V17"), model.predict, X_test, ice=True, grid_resolution=20)
    plt.show()

