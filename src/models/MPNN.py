import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, global_add_pool


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8):
        """
        Initialize a Graph Attention Layer (GAT).

        Parameters:
        in_channels (int): Number of input features for each node.
        out_channels (int): Number of output features per head.
        heads (int): Number of attention heads.
        """
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads)

    def forward(self, x, edge_index):
        """
        Forward pass through the GAT layer.

        Parameters:
        x (torch.Tensor): Node feature matrix.
        edge_index (torch.Tensor): Graph connectivity in COO format.

        Returns:
        torch.Tensor: Output feature matrix after applying the GAT layer.
        """
        return self.conv(x, edge_index)


class MPNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        """
        Initialize the Message Passing Neural Network (MPNN) model.

        Parameters:
        in_channels (int): Number of input features for each node.
        hidden_channels (int): Number of hidden features in GNN layers.
        out_channels (int): Number of output features.
        num_layers (int): Number of GNN layers.
        """
        super(MPNNModel, self).__init__()
        self.layers = nn.ModuleList()

        # Input GCN layer
        self.layers.append(GCNConv(in_channels, hidden_channels))

        # Hidden GCN layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))

        # Output GCN layer
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the MPNN.

        Parameters:
        x (torch.Tensor): Node feature matrix.
        edge_index (torch.Tensor): Graph connectivity in COO format.
        batch (torch.Tensor): Batch vector, which assigns each node to a specific graph.

        Returns:
        torch.Tensor: Graph-level output after global pooling.
        """
        # Apply each layer with ReLU activation for intermediate layers
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x, edge_index))

        # Apply the output layer without activation for regression
        x = self.layers[-1](x, edge_index)

        # Apply global pooling to obtain graph-level representation
        x = global_add_pool(x, batch)

        return x


if __name__ == "__main__":
    # Example usage of MPNN model
    in_channels = 10
    hidden_channels = 32
    out_channels = 1
    num_layers = 4

    model = MPNNModel(in_channels, hidden_channels, out_channels, num_layers)
    example_x = torch.rand((100, in_channels))  # Example node features for 100 nodes
    example_edge_index = torch.randint(0, 100, (2, 200))  # Example edges in COO format
    example_batch = torch.zeros(100, dtype=torch.long)  # Assign all nodes to the same graph for simplicity

    output = model(example_x, example_edge_index, example_batch)
    print("Graph-level output:", output)
