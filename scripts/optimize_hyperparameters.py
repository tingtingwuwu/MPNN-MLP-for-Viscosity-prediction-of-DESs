import pandas as pd
import torch
from src.data_processing import load_and_preprocess_data
from src.training.hyperparameter_optimization import optimize_hyperparameters

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    """
    Main execution function for data preprocessing and hyperparameter optimization.

    This function loads the dataset, preprocesses the features, and runs hyperparameter optimization.
    """
    # Specify the path to the data file (hidden for confidentiality)
    file_path = "<path_to_data_file>"
    feature_cols = ['feature1', 'feature2', 'feature3']  # Adjust based on the actual feature columns in your dataset
    target_col = 'Viscosity, cP'

    # Load and preprocess the dataset
    numeric_features, target = load_and_preprocess_data(file_path, feature_cols, target_col)

    # Placeholder for processed SMILES features (e.g., molecular representations)
    features_smiles1 = None  # TODO: Replace with actual SMILES processing output
    features_smiles2 = None  # TODO: Replace with actual SMILES processing output

    # Perform hyperparameter optimization
    best_params = optimize_hyperparameters(numeric_features=numeric_features, target=target,
                                           features_smiles1=features_smiles1, features_smiles2=features_smiles2,
                                           device=device)
    print("Best hyperparameters found:", best_params)


if __name__ == "__main__":
    main()
