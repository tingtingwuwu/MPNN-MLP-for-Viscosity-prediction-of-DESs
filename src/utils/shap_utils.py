import shap
import torch
import numpy as np
from src.training.train_gnn import extract_features_and_cache

def model_predict(smiles1, smiles2, numeric_features, mpnn_model, mlp_model, device):
    """
    Generate model predictions based on the given SMILES strings and numeric features.

    This function extracts features from the SMILES strings using the given GNN model, combines them with
    additional numeric features, and then passes them through an MLP model to generate predictions.

    Parameters:
    smiles1 (list of str): List of SMILES strings for the first component.
    smiles2 (list of str): List of SMILES strings for the second component.
    numeric_features (torch.Tensor): Tensor of numeric features to be combined.
    mpnn_model (torch.nn.Module): The trained GNN model used for feature extraction.
    mlp_model (torch.nn.Module): The trained MLP model used for prediction.
    device (str): Device to perform computations ('cpu' or 'cuda').

    Returns:
    np.ndarray: Predicted values as a NumPy array.
    """
    with torch.no_grad():
        # Extract features from SMILES strings using the GNN model
        features_smiles1, _, _, _ = extract_features_and_cache(mpnn_model, smiles1, device)
        features_smiles2, _, _, _ = extract_features_and_cache(mpnn_model, smiles2, device)

        # Handle different batch sizes between SMILES features and numeric features
        if numeric_features.size(0) == 1:
            smiles1_repeated = features_smiles1[0].unsqueeze(0)
            smiles2_repeated = features_smiles2[0].unsqueeze(0)
        else:
            repeat_factor = numeric_features.size(0) // features_smiles1.size(0)
            smiles1_repeated = features_smiles1.repeat(repeat_factor, 1)
            smiles2_repeated = features_smiles2.repeat(repeat_factor, 1)

        # Combine all features for final prediction
        combined_features = torch.cat((smiles1_repeated, smiles2_repeated, numeric_features), dim=1)
        output = mlp_model(combined_features)
    return output.cpu().numpy()

def compute_shap_values(smiles1, smiles2, numeric_features, mpnn_model, mlp_model, device, k=50, nsamples=100):
    """
    Compute SHAP values for a given model using SMILES and numeric features.

    This function uses KernelExplainer to estimate the SHAP values, which indicate the impact of each feature on
    the model's predictions, helping to understand the model's decision-making process.

    Parameters:
    smiles1 (list of str): List of SMILES strings for the first component.
    smiles2 (list of str): List of SMILES strings for the second component.
    numeric_features (torch.Tensor): Tensor of numeric features to be explained.
    mpnn_model (torch.nn.Module): The trained GNN model used for feature extraction.
    mlp_model (torch.nn.Module): The trained MLP model used for prediction.
    device (str): Device to perform computations ('cpu' or 'cuda').
    k (int): Number of clusters for k-means to use as background data for SHAP (default is 50).
    nsamples (int): Number of samples to use for estimating SHAP values (default is 100).

    Returns:
    np.ndarray: SHAP values for the numeric features.
    """
    # Convert numeric features to NumPy format for SHAP
    numeric_features_np = numeric_features.cpu().numpy()

    # Use k-means clustering to generate background data for SHAP
    background = shap.kmeans(numeric_features_np, k)

    # Select SMILES strings for the background set
    smiles1_background = smiles1[:k]
    smiles2_background = smiles2[:k]

    # Define the SHAP KernelExplainer
    explainer = shap.KernelExplainer(
        lambda X: model_predict(smiles1_background, smiles2_background,
                                torch.tensor(X, dtype=torch.float32, device=device), mpnn_model, mlp_model, device),
        background.data
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(numeric_features_np, nsamples=nsamples)
    return shap_values

if __name__ == "__main__":
    # Example usage (requires appropriate models and data)
    try:
        # Placeholder data
        smiles1 = ["CCO", "C1=CC=CC=C1"]
        smiles2 = ["CCO", "C1=CC=CC=C1"]
        numeric_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        device = 'cpu'

        # Placeholder for actual model instances
        mpnn_model = None  # Replace with actual trained model
        mlp_model = None  # Replace with actual trained model

        # Compute SHAP values
        shap_values = compute_shap_values(smiles1, smiles2, numeric_features, mpnn_model, mlp_model, device)
        print("SHAP values computed successfully:", shap_values)
    except Exception as e:
        print(f"An error occurred during SHAP computation: {e}")
