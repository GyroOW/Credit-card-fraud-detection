import shap
import matplotlib.pyplot as plt

def interpret_model(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar')
    plt.show()

    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot("V14", model.predict, X_test)
    plt.show()
    
    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot("V17", model.predict, X_test)
    plt.show()
    
    plt.figure(figsize=(14, 10))
    shap.partial_dependence_plot(("V14", "V17"), model.predict, X_test, ice=True, grid_resolution=20)
    plt.show()
