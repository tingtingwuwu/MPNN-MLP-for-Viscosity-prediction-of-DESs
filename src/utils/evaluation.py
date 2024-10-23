import shap
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate common evaluation metrics for regression models.

    This function calculates the Mean Squared Error (MSE) and R-squared (R²) score between
    the true and predicted values to assess model performance.

    Parameters:
    y_true (array-like): Array of true target values.
    y_pred (array-like): Array of predicted target values.

    Returns:
    dict: Dictionary containing the MSE and R² scores.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "r2": r2}

def compute_shap_values(model, data_loader, num_samples=100):
    """
    Compute SHAP values for a given model using a batch of data from the data loader.

    This function provides insights into the contribution of each feature to the model's output
    by utilizing SHAP KernelExplainer on a subset of the input data.

    Parameters:
    model (torch.nn.Module): Trained model to be explained.
    data_loader (DataLoader): Data loader providing the input data.
    num_samples (int): Number of samples to use for SHAP analysis (default is 100).

    Returns:
    np.ndarray: SHAP values representing the impact of each input feature on the model's predictions.
    """
    model.eval()

    # Select a batch of data from the data loader
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        raise ValueError("The data loader is empty. Please provide valid input data.")

    X = batch[0]  # Assuming batch contains (inputs, targets)
    if num_samples < len(X):
        X = X[:num_samples]

    # Convert data to NumPy array for SHAP compatibility
    X = X.cpu().detach().numpy()

    # Use KernelExplainer to compute SHAP values
    explainer = shap.KernelExplainer(model_predict, X)
    shap_values = explainer.shap_values(X)

    return np.array(shap_values)

def model_predict(input_data):
    """
    Model prediction function to be used with SHAP explanations.

    This function converts input data into a torch tensor, performs model inference,
    and returns the predictions in NumPy format.

    Parameters:
    input_data (np.ndarray): Input data for the model predictions.

    Returns:
    np.ndarray: Predicted output values from the model.
    """
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Ensure that the model is defined globally
    if 'model' not in globals():
        raise RuntimeError("The model must be defined globally before using 'model_predict'.")

    # Perform inference
    with torch.no_grad():
        predictions = model(input_tensor).cpu().numpy()
    return predictions

if __name__ == "__main__":
    # Example usage for calculate_metrics
    y_true = [3.0, 2.5, 4.0, 7.0]
    y_pred = [2.8, 2.7, 3.9, 7.2]
    metrics = calculate_metrics(y_true, y_pred)
    print("Evaluation Metrics:", metrics)

    # Example usage for compute_shap_values (placeholder)
    try:
        # Note: The following requires an actual model and data_loader instance
        shap_values = compute_shap_values(model, data_loader, num_samples=50)
        print("SHAP values computed successfully.")
    except Exception as e:
        print(f"Error during SHAP computation: {e}")
