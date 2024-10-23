from sklearn.metrics import mean_squared_error, r2_score
from src.evaluation.calculate_metrics import calculate_aard
from src.models.TraditionalMLModels import get_traditional_ml_model
import pickle


def train_and_evaluate_traditional_ml(model_name, X_train, y_train, X_val, y_val, **kwargs):
    """
    Train and evaluate a traditional machine learning model.

    This function initializes a traditional machine learning model, trains it on the provided training data,
    and evaluates it on the validation set using Mean Squared Error (MSE), R-squared (R²), and Average Absolute
    Relative Deviation (AARD) metrics.

    Parameters:
    model_name (str): The name of the model to train ('RandomForest', 'SVR', 'LinearRegression', etc.).
    X_train (array-like): Training features.
    y_train (array-like): Training targets.
    X_val (array-like): Validation features.
    y_val (array-like): Validation targets.
    kwargs: Additional keyword arguments for model initialization, such as hyperparameters.

    Returns:
    dict: A dictionary containing the evaluation metrics (MSE, R², AARD).
    """
    # Initialize the model based on the model name
    model = get_traditional_ml_model(model_name, **kwargs)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    aard = calculate_aard(y_val, y_pred)

    # Output evaluation results
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation R²: {r2:.4f}")
    print(f"Validation AARD: {aard:.2f}%")

    # Return evaluation metrics as a dictionary
    metrics = {
        'mse': mse,
        'r2': r2,
        'aard': aard
    }

    return metrics


if __name__ == "__main__":
    # Example usage (this requires properly defined data)
    try:
        # Placeholder data for demonstration purposes
        X_train, y_train, X_val, y_val = [[1, 2], [3, 4]], [5, 6], [[7, 8]], [9]

        metrics = train_and_evaluate_traditional_ml('RandomForest', X_train, y_train, X_val, y_val, n_estimators=100)
        print("Evaluation Metrics:", metrics)
    except Exception as e:
        print(f"An error occurred during model training and evaluation: {e}")
