import shap
import numpy as np
import torch

def compute_shap_values(model, data_loader, num_samples=100):
    """
    Compute SHAP values for the given model and input data.

    Parameters:
    model (torch.nn.Module): The trained model to be explained.
    data_loader (DataLoader): DataLoader providing the input data.
    num_samples (int): Number of samples to be used for SHAP analysis.

    Returns:
    np.ndarray: SHAP values for the input features.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Select a batch of data to explain
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        raise ValueError("The data loader is empty. Please provide valid input data.")

    X = batch[0]  # Assuming batch contains (inputs, targets)
    if num_samples < len(X):
        X = X[:num_samples]  # Limit to the specified number of samples if needed

    # Convert data to NumPy for SHAP compatibility
    X = X.cpu().detach().numpy()

    # Use KernelExplainer for model-agnostic explanations
    explainer = shap.KernelExplainer(model_predict, X)
    shap_values = explainer.shap_values(X)

    return np.array(shap_values)

def model_predict(input_data):
    """
    Prediction function wrapper for SHAP.

    Parameters:
    input_data (np.ndarray): Input data in NumPy array format.

    Returns:
    np.ndarray: Model predictions for the input data.
    """
    # Convert input data to a torch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Ensure the model is defined in the global scope for use within this function
    if 'model' not in globals():
        raise RuntimeError("Model must be defined in the global scope before calling model_predict.")

    # Use the model to make predictions
    with torch.no_grad():
        predictions = model(input_tensor).cpu().numpy()

    return predictions

if __name__ == "__main__":
    # Example placeholder usage to illustrate how to call these functions
    # Note: This assumes that the `model` and `data_loader` are defined elsewhere in the codebase.
    try:
        shap_values = compute_shap_values(model, data_loader, num_samples=50)
        print("SHAP values computed successfully.")
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
