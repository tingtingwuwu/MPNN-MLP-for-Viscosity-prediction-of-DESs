import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from src.training.checkpoint_utils import save_checkpoint
from src.evaluation.calculate_metrics import calculate_mse, calculate_r2
from src.utils.logging_utils import log_training_info

def train_and_evaluate_gnn(model, train_loader, val_loader, epochs, learning_rate, device='cpu'):
    """
    Train and evaluate a Graph Neural Network (GNN) model.

    Parameters:
    model (torch.nn.Module): The GNN model to be trained and evaluated.
    train_loader (DataLoader): DataLoader providing the training data.
    val_loader (DataLoader): DataLoader providing the validation data.
    epochs (int): Number of training epochs.
    learning_rate (float): Learning rate for the optimizer.
    device (str): Device to train the model on ('cpu' or 'cuda').

    Returns:
    float: The average validation loss after training.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        log_training_info(epoch, avg_loss)

        # Validation phase
        model.eval()
        y_true, y_pred = [], []
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                val_loss = criterion(out, data.y)
                total_val_loss += val_loss.item()
                y_true.extend(data.y.tolist())
                y_pred.extend(out.tolist())

        avg_val_loss = total_val_loss / len(val_loader)

        # Calculate evaluation metrics for validation
        mse = calculate_mse(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)
        log_training_info(epoch, avg_loss, mse=mse, r2=r2)

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, filename=f"checkpoint_epoch_{epoch}.pth")

    return avg_val_loss

if __name__ == "__main__":
    # Example placeholder usage (this requires defined DataLoaders and a model)
    try:
        # Placeholder for demonstration; replace train_loader, val_loader, and model with actual instances
        train_loader = DataLoader([])
        val_loader = DataLoader([])
        model = None  # Replace with an actual model instance
        avg_val_loss = train_and_evaluate_gnn(model, train_loader, val_loader, epochs=10, learning_rate=0.001, device='cuda')
        print(f"Training completed. Average validation loss: {avg_val_loss:.4f}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
