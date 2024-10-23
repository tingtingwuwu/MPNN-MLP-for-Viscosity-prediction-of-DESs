import optuna
from src.models.MLP import MLPModel
from src.models.MPNN import MPNNModel
from src.training.train_gnn import train_and_evaluate_gnn
from src.training.train_traditional_ml import train_and_evaluate_traditional_ml

def optimize_hyperparameters(train_loader, val_loader, in_channels, out_channels, device='cpu'):
    """
    Optimize hyperparameters for both GNN and MLP models using Optuna.

    This function defines an Optuna study to find the best hyperparameters for either a GNN or an MLP model by minimizing
    the validation loss. The objective function includes hidden layer dimensions, number of layers, learning rate, and model type.

    Parameters:
    train_loader (DataLoader): DataLoader for training data.
    val_loader (DataLoader): DataLoader for validation data.
    in_channels (int): Number of input features for the model.
    out_channels (int): Number of output features for the model.
    device (str): Device on which to train the model ('cpu' or 'cuda').
    """
    def objective(trial):
        """
        Objective function for the Optuna study.

        Parameters:
        trial (optuna.Trial): A trial object to suggest hyperparameters.

        Returns:
        float: Validation loss for the selected model and hyperparameters.
        """
        # Suggest hyperparameters
        hidden_channels = trial.suggest_int("hidden_channels", 16, 128)
        num_layers = trial.suggest_int("num_layers", 2, 6)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        model_type = trial.suggest_categorical("model_type", ["MLP", "GNN"])

        # Initialize model based on selected trial parameters
        if model_type == "MLP":
            model = MLPModel(in_channels, hidden_channels, out_channels, num_layers).to(device)
            val_loss = train_and_evaluate_traditional_ml(model, X_train, y_train, X_val, y_val, learning_rate=learning_rate)
        else:
            model = MPNNModel(in_channels, hidden_channels, out_channels, num_layers).to(device)
            val_loss = train_and_evaluate_gnn(model, train_loader, val_loader, epochs=5, learning_rate=learning_rate, device=device)

        return val_loss

    # Create an Optuna study and start the optimization process
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # Print the best trial parameters found during the optimization
    print("Best trial parameters:", study.best_trial.params)

if __name__ == "__main__":
    # Example usage (this requires properly defined DataLoaders and data)
    try:
        optimize_hyperparameters(train_loader, val_loader, in_channels=10, out_channels=1, device='cuda')
    except Exception as e:
        print(f"An error occurred during hyperparameter optimization: {e}")
