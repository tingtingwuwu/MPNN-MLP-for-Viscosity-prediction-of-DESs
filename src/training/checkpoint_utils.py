import torch

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    """
    Save a model checkpoint to a file.

    Parameters:
    model (torch.nn.Module): The model to be saved.
    optimizer (torch.optim.Optimizer): The optimizer used during training, whose state will also be saved.
    epoch (int): The current epoch number to save for potential resumption.
    filename (str): The file name for the checkpoint (default is 'checkpoint.pth').
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved successfully at '{filename}'.")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """
    Load a model checkpoint from a file.

    Parameters:
    model (torch.nn.Module): The model instance to load the weights into.
    optimizer (torch.optim.Optimizer): The optimizer instance to load the state into.
    filename (str): The file name for the checkpoint to be loaded (default is 'checkpoint.pth').

    Returns:
    int: The epoch number from which training can be resumed.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded successfully from '{filename}', resuming from epoch {epoch}.")
    return epoch

if __name__ == "__main__":
    # Example usage for saving and loading checkpoints
    from torch import nn, optim

    # Define a simple model and optimizer for demonstration purposes
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Save a checkpoint
    save_checkpoint(model, optimizer, epoch=5, filename="example_checkpoint.pth")

    # Load the checkpoint
    loaded_epoch = load_checkpoint(model, optimizer, filename="example_checkpoint.pth")
    print(f"Model and optimizer state loaded. Resuming from epoch {loaded_epoch}.")
