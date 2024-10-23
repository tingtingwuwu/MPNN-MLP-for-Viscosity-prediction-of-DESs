from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


def get_traditional_ml_model(model_name, **kwargs):
    """
    Retrieve a traditional machine learning model based on the specified model name.

    Parameters:
    model_name (str): The name of the model to retrieve. Options are 'RandomForest', 'SVR', or 'LinearRegression'.
    kwargs: Additional keyword arguments for model initialization, allowing for customization of hyperparameters.

    Returns:
    sklearn.base.BaseEstimator: An initialized instance of the selected machine learning model.

    Raises:
    ValueError: If the specified model name is not recognized.
    """
    if model_name == 'RandomForest':
        return RandomForestRegressor(**kwargs)
    elif model_name == 'SVR':
        return SVR(**kwargs)
    elif model_name == 'LinearRegression':
        return LinearRegression(**kwargs)
    else:
        raise ValueError(
            f"Model '{model_name}' not recognized. Please choose from 'RandomForest', 'SVR', or 'LinearRegression'.")


if __name__ == "__main__":
    # Example usage of get_traditional_ml_model
    try:
        model = get_traditional_ml_model('RandomForest', n_estimators=100, random_state=42)
        print("Initialized model:", model)
    except ValueError as e:
        print(e)
