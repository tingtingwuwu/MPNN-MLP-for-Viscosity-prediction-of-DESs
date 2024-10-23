import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors


def load_and_preprocess_data(file_path, feature_cols, target_col):
    """
    Load and preprocess data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    feature_cols (list of str): List of feature column names to be used as predictors.
    target_col (str): Name of the target column.

    Returns:
    pd.DataFrame, pd.Series: Preprocessed feature matrix and target variable.
    """
    # Load data from CSV
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The specified file was not found at the given path: {file_path}")

    # Handle missing values by filling with the mean value
    data.fillna(data.mean(), inplace=True)

    # Select features and target
    try:
        features = data[feature_cols]
        target = data[target_col]
    except KeyError as e:
        raise KeyError(f"Error: One or more specified columns were not found in the dataset. Details: {e}")

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(features)

    return numeric_features, target


if __name__ == "__main__":
    # Example usage (file path is hidden for confidentiality)
    file_path = "<path_to_data_file>"
    feature_cols = ['feature1', 'feature2', 'feature3']  # Adjust to the actual feature columns in your dataset
    target_col = 'Viscosity, cP'

    try:
        numeric_features, target = load_and_preprocess_data(file_path, feature_cols, target_col)
        print("Feature Matrix Shape:", numeric_features.shape)
        print("Target Variable Shape:", target.shape)
    except (FileNotFoundError, KeyError) as e:
        print(e)
