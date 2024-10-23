from rdkit import Chem

def smiles_to_molecule(smiles):
    """
    Convert a SMILES string to an RDKit molecule.

    This function takes a SMILES (Simplified Molecular Input Line Entry System) string and converts
    it into an RDKit molecule object, which can be used for further molecular manipulations and analysis.

    Parameters:
    smiles (str): The SMILES representation of a molecule.

    Returns:
    Mol: An RDKit Mol object representing the molecule. Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return mol

def molecule_to_smiles(mol):
    """
    Convert an RDKit molecule to a canonical SMILES string.

    This function converts an RDKit molecule object back to its SMILES representation, ensuring a
    consistent and unique canonical SMILES string for the given molecule.

    Parameters:
    mol (Mol): An RDKit Mol object representing the molecule.

    Returns:
    str: A canonical SMILES representation of the molecule.
    """
    if mol is None:
        raise ValueError("Invalid Mol object: The molecule cannot be None.")
    return Chem.MolToSmiles(mol)

if __name__ == "__main__":
    # Example usage for smiles_to_molecule and molecule_to_smiles
    try:
        smiles = "CCO"  # Ethanol
        mol = smiles_to_molecule(smiles)
        print(f"RDKit Mol object created for SMILES '{smiles}':", mol)

        canonical_smiles = molecule_to_smiles(mol)
        print(f"Canonical SMILES representation: {canonical_smiles}")
    except ValueError as e:
        print(e)
