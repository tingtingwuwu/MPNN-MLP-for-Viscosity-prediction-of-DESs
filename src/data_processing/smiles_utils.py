import torch
import networkx as nx
from rdkit import Chem
from torch_geometric.utils import from_networkx

def atom_features(atom):
    """
    Extract features for an atom.

    Parameters:
    atom (rdkit.Chem.rdchem.Atom): RDKit Atom object.

    Returns:
    list: A list of atom-level features, including atomic number, degree, hybridization, aromaticity, hydrogen count, valence, and formal charge.
    """
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetTotalNumHs(),
        atom.GetExplicitValence(),
        atom.GetFormalCharge()
    ]

def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a graph representation suitable for input to a Graph Neural Network (GNN).

    Parameters:
    smiles (str): SMILES representation of the molecule.

    Returns:
    torch_geometric.data.Data: Graph data representation of the molecule, containing atom features and bond information.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string encountered: {smiles}")

    # Create a NetworkX graph from the RDKit molecule
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), x=torch.tensor(atom_features(atom), dtype=torch.float))

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # Convert the NetworkX graph to a PyTorch Geometric graph
    data = from_networkx(G)
    return data

if __name__ == "__main__":
    # Example usage (with a placeholder SMILES string for testing purposes)
    smiles = "CCO"  # Ethanol, as an example
    try:
        graph_data = smiles_to_graph(smiles)
        print("Graph representation:", graph_data)
    except ValueError as e:
        print(e)
