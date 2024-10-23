import os
import json


def ensure_directory_exists(directory):
    """
    Ensure that the specified directory exists. If it does not exist, create it.

    This function checks if a given directory path exists and creates it if it does not,
    ensuring that subsequent operations involving this directory will not fail due to
    missing directories.

    Parameters:
    directory (str): The directory path to ensure exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_json(data, file_path):
    """
    Save a dictionary to a JSON file.

    This function ensures the directory for the file path exists, then saves the given
    dictionary as a JSON file with proper indentation for readability.

    Parameters:
    data (dict): The data to be saved in JSON format.
    file_path (str): The path where the JSON file will be saved.
    """
    # Ensure the directory exists before saving the JSON file
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data successfully saved to {file_path}")


def load_json(file_path):
    """
    Load a dictionary from a JSON file.

    This function reads data from a JSON file and returns it as a Python dictionary.

    Parameters:
    file_path (str): The path to the JSON file to be loaded.

    Returns:
    dict: The data loaded from the JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data


if __name__ == "__main__":
    # Example usage for saving and loading JSON
    try:
        data_to_save = {"key": "value", "number": 123}
        save_path = "./example_dir/example_file.json"
        save_json(data_to_save, save_path)
        loaded_data = load_json(save_path)
        print("Loaded Data:", loaded_data)
    except FileNotFoundError as e:
        print(e)
