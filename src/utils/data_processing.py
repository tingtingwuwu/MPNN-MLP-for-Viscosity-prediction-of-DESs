import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors


def load_and_preprocess_data(file_path, feature_cols, target_col):
    """
    Load and preprocess data from a CSV file.

    This function reads data from a specified CSV file, fills any missing values with the column mean,
    scales the features using `StandardScaler`, and returns the processed features and target.

    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    feature_cols (list of str): List of feature column names to be used as predictors.
    target_col (str): Name of the target column to be predicted.

    Returns:
    pd.DataFrame, pd.Series: The preprocessed feature matrix and the target variable.
    """
    # Load data from CSV file
    data = pd.read_csv(file_path)

    # Handle missing values by filling with mean values
    data.fillna(data.mean(), inplace=True)

    # Extract features and target
    features = data[feature_cols]
    target = data[target_col]

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, target


def calculate_descriptors(smiles_list):
    """
    Calculate molecular descriptors for a given list of SMILES strings.

    This function calculates a set of molecular descriptors, including molecular weight, topological
    polar surface area (TPSA), and partition coefficient (MolLogP), for each SMILES string provided.

    Parameters:
    smiles_list (list of str): List of SMILES strings representing molecular structures.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated descriptors for each SMILES string.
    """
    descriptors = [Descriptors.MolWt, Descriptors.TPSA, Descriptors.MolLogP]
    data = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string encountered: {smiles}")
        descriptor_values = [desc(mol) for desc in descriptors]
        data.append(descriptor_values)

    return pd.DataFrame(data, columns=[desc.__name__ for desc in descriptors])


if __name__ == "__main__":
    # Example usage for load_and_preprocess_data
    try:
        file_path = "<path_to_data_file>"
        feature_cols = ['feature1', 'feature2', 'feature3']  # Adjust as per the actual feature columns
        target_col = 'Viscosity, cP'
        features, target = load_and_preprocess_data(file_path, feature_cols, target_col)
        print("Feature matrix shape:", features.shape)
        print("Target shape:", target.shape)
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    # Example usage for calculate_descriptors
    try:
        smiles_list = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]  # Replace with actual SMILES strings
        descriptor_df = calculate_descriptors(smiles_list)
        print(descriptor_df)
    except ValueError as e:
        print(e)
