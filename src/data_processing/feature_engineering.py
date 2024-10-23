from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors, rdMolDescriptors
def calculate_descriptors(smiles_list):
    """
    Calculate molecular descriptors for a given list of SMILES strings.

    Parameters:
    smiles_list (list of str): List of SMILES strings.

    Returns:
    pd.DataFrame: DataFrame containing calculated molecular descriptors for each SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles_list)
    if mol is None:
        return None

    descriptors = {
        "ExactMolWt": Descriptors.ExactMolWt(mol),  # Exact molecular weight
        "MolWt": Descriptors.MolWt(mol),  # Molecular weight
        "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt(mol),  # Heavy atom molecular weight
        "Chi0": Descriptors.Chi0(mol),  # Chi0 connectivity descriptor
        "Chi1": Descriptors.Chi1(mol),  # Chi1 connectivity descriptor
        "Chi2n": Descriptors.Chi2n(mol),  # Chi2n connectivity descriptor
        "Chi3v": Descriptors.Chi3v(mol),  # Chi3v connectivity descriptor
        "Chi4v": Descriptors.Chi4v(mol),  # Chi4v connectivity descriptor
        "Kappa1": Descriptors.Kappa1(mol),  # Topological descriptor Kappa1
        "Kappa2": Descriptors.Kappa2(mol),  # Topological descriptor Kappa2
        "Kappa3": Descriptors.Kappa3(mol),  # Topological descriptor Kappa3
        "BalabanJ": Descriptors.BalabanJ(mol),  # BalabanJ topological index
        "LabuteASA": Descriptors.LabuteASA(mol),  # Labute accessible surface area
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(mol),  # Maximum partial charge
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),  # Minimum partial charge
        "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex(mol),  # Maximum electronic state value
        "MinAbsEStateIndex": Descriptors.MinAbsEStateIndex(mol),  # Minimum electronic state value
        "MolLogP": Descriptors.MolLogP(mol),  # LogP
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),  # Number of hydrogen bond acceptors
        "NumHDonors": Descriptors.NumHDonors(mol),  # Number of hydrogen bond donors
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),  # Number of rotatable bonds
        "FractionCSP3": Descriptors.FractionCSP3(mol),  # Fraction of saturated carbons (CSP3)
        "RingCount": rdMolDescriptors.CalcNumRings(mol),  # Number of rings
        "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms(mol),  # Number of heavy atoms
        "PMI1": Descriptors.PMI1(mol),  # First principal moment of inertia
        "PMI2": Descriptors.PMI2(mol),  # Second principal moment of inertia
        "PMI3": Descriptors.PMI3(mol)  # Third principal moment of inertia
    }
    return descriptors

    descriptors_data = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string encountered: {smiles}")

        # Calculate descriptor values for the molecule
        descriptor_values = [desc(mol) for desc in descriptor_list]
        descriptors_data.append(descriptor_values)

    # Create a DataFrame with calculated descriptors
    descriptor_df = pd.DataFrame(descriptors_data, columns=[desc.__name__ for desc in descriptor_list])
    return descriptor_df

if __name__ == "__main__":
    # Example usage (with a placeholder list of SMILES strings for testing purposes)
    smiles_list = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]  # Replace with actual SMILES strings
    try:
        descriptor_df = calculate_descriptors(smiles_list)
        print(descriptor_df)
    except ValueError as e:
        print(e)
