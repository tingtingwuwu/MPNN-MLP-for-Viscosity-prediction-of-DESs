import torch
import pandas as pd
import numpy as np
from src.data_processing import load_and_preprocess_data
from src.models.gnn import MPNNModel
from src.models.ml_models import models
from src.training.train_gnn import train_and_evaluate_gnn_mlp
from src.training.train_ml import train_and_evaluate_ml
from src.utils.shap_utils import compute_shap_values
from src.utils.feature_extraction import extract_features_and_cache

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    """
    Main function to load data, train models, and compute SHAP values for interpretability.
    """
    # Specify the path to the data file (hidden for confidentiality)
    file_path = "<path_to_data_file>"
    feature_cols = ['feature1', 'feature2', 'feature3']  # Adjust based on the actual feature columns in your dataset
    target_col = 'Viscosity, cP'

    # Load and preprocess the dataset
    numeric_features, target = load_and_preprocess_data(file_path, feature_cols, target_col)

    # Train and evaluate GNN + MLP models
    mpnn_model = MPNNModel(in_channels=8, hidden_channels=64, out_channels=32, num_layers=6).to(device)
    mlp_model = MLPModel(input_dim=numeric_features.shape[1], hidden_dim=128, output_dim=1, num_layers=3).to(device)

    r2_gnn, mse_gnn, aard_gnn = train_and_evaluate_gnn_mlp(mpnn_model, mlp_model, None, None, numeric_features, target,
                                                           device)

    # Train and evaluate traditional machine learning models
    for model_name, model in models.items():
        r2, mse, aard = train_and_evaluate_ml(model, numeric_features, target)
        print(f"Model: {model_name} - R²: {r2:.3f}, MSE: {mse:.3f}, AARD: {aard:.3f}")

    # Extract and cache features from Component #1
    print("Extracting and caching features for Component #1...")
    features_smiles1, valid_smiles1, invalid_indices1, valid_indices1 = extract_features_and_cache(mpnn_model,
                                                                                                   smiles_list1, device)

    # Extract and cache features from Component #2
    print("Extracting and caching features for Component #2...")
    features_smiles2, valid_smiles2, invalid_indices2, valid_indices2 = extract_features_and_cache(mpnn_model,
                                                                                                   smiles_list2, device)

    # Filter numeric features and target using valid indices from both components
    valid_indices = list(set(valid_indices1).intersection(valid_indices2))
    numeric_features = numeric_features.iloc[valid_indices]
    target = target.iloc[valid_indices]

    # Compute SHAP values for interpretability
    numeric_features_tensor = torch.tensor(numeric_features.values, dtype=torch.float32, device=device)
    shap_values = compute_shap_values(valid_smiles1, valid_smiles2, numeric_features_tensor, mpnn_model, mlp_model,
                                      device)

    # Print or save SHAP values
    shap_values_np = np.array(shap_values)
    print("SHAP values:", shap_values_np)

    # Save SHAP values to a CSV file
    shap_values_2d = shap_values_np[0]
    shap_df = pd.DataFrame(shap_values_2d)
    shap_df.to_csv("shap_values.csv", index=False)
    print("SHAP values saved to shap_values.csv")


if __name__ == "__main__":
    main()
