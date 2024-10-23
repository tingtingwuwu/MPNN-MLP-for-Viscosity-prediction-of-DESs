from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The Mean Squared Error between `y_true` and `y_pred`.
    """
    return mean_squared_error(y_true, y_pred)


def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared (R²) score to measure the proportion of variance explained by the model.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The R² score between `y_true` and `y_pred`.
    """
    return r2_score(y_true, y_pred)


def calculate_aard(y_true, y_pred):
    """
    Calculate Average Absolute Relative Deviation (AARD) between true and predicted values.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The AARD value as a percentage, indicating the average relative prediction error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    relative_errors = np.abs((y_true - y_pred) / y_true)
    return np.mean(relative_errors) * 100  # Convert to percentage form


if __name__ == "__main__":
    # Example usage with placeholder values for testing purposes
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.1, 7.8]

    mse = calculate_mse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    aard = calculate_aard(y_true, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Average Absolute Relative Deviation (AARD): {aard:.2f}%")
