import torch

def initialize_weights(model):
    """
    Initialize the weights of the model using Xavier initialization.

    This function iterates through all layers of the given model and applies Xavier uniform initialization to the weights
    of `torch.nn.Linear` and `torch.nn.Conv2d` layers. Additionally, biases are initialized to zero if they are present.

    Parameters:
    model (torch.nn.Module): The model whose weights need to be initialized.
    """
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

if __name__ == "__main__":
    # Example usage of initialize_weights
    from torch.nn import Sequential, Linear, ReLU

    # Define a simple sequential model for demonstration purposes
    example_model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 1)
    )

    # Initialize the model weights
    initialize_weights(example_model)
    print("Weights initialized using Xavier initialization.")
