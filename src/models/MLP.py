import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        """
        Initialize the Multi-Layer Perceptron (MLP) model.

        Parameters:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of neurons in each hidden layer.
        output_dim (int): Number of output features.
        num_layers (int): Total number of layers in the model, including input and output layers.
        """
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        """
        Perform a forward pass through the MLP model.

        Parameters:
        x (torch.Tensor): Input data tensor.

        Returns:
        torch.Tensor: Output predictions from the model.
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


if __name__ == "__main__":
    # Example usage of the MLP model
    input_dim = 10
    hidden_dim = 32
    output_dim = 1
    num_layers = 4

    model = MLPModel(input_dim, hidden_dim, output_dim, num_layers)
    example_input = torch.rand((5, input_dim))  # Batch size of 5
    output = model(example_input)
    print("Model output:", output)
