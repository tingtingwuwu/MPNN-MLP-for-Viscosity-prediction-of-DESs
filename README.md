#Multiscale exploration of informative latent features for accurate deep eutectic solvents viscosity prediction

## Overview
Deep eutectic solvents (DESs) are promising green solvents widely used in chemical processes, separations, and catalysis, where viscosity plays a critical role in their industrial feasibility. However, measuring DESs viscosity is time-consuming and labor-intensive due to numerous influencing factors. In this work, we propose a Message Passing Neural Network (MPNN)-Graph Attention Networks (GAT)-multilayer perceptron (MLP) framework for developing DESs viscosity prediction model through end-to-end, data-driven training, emphasizing the implicit extraction and embedding of effective features. Specifically, a dataset comprising 5790 DESs and their corresponding experimental viscosity values was compiled from published literature. Recognizing the essential role of SMILES in predicting DESs viscosity, two stacked GAT were utilized to implicitly capture interdependencies among molecular substructures, enabling the extraction of significant features. Given that DESs are typically binary systems, the predicted density is incorporated as an additional input, reducing reliance on experimental data and enhancing predictive accuracy. Finally, a MLP is used to integrate these extracted features with raw physical and chemical properties, effectively assisting in multiscale training and prediction. Comprehensive experiments demonstrate that this framework significantly improves the accuracy of DESs viscosity predictions, paving the way for more efficient and sustainable green chemistry innovations.

## Prerequisites

### System Information
- **Operating System**: Windows 10 (version 19045)
- **Processor**: Intel(R) Core(TM) i7-11800H CPU @ 2.30GHz, featuring 14 physical cores and 20 logical processors, with a maximum CPU frequency of 3.5GHz.
- **Graphics Card**: NVIDIA GeForce RTX 4060 Ti with 16380MB of dedicated VRAM.

### Key Library Versions
- **Python Version**: 3.9.19
- **PyTorch**: 2.3.0
- **Pandas**: 2.2.3
- **NumPy**: 1.26.4
- **Scikit-learn**: 1.0.2
- **Matplotlib**: 3.9.2
- **NetworkX**: 3.2.1
- **RDKit**: 2024.03.5
- **SHAP**: 0.31.0

### Dependencies
- Install all required dependencies by running the following command:

  ```bash
  pip install -r requirements.txt
  ```
  The key packages used in this project include:
  - `numpy` and `pandas` for data manipulation
  - `scikit-learn` for preprocessing and evaluation metrics
  - `torch` and `torch-geometric` for neural network operations
  - `rdkit` for handling chemical structures
  - `shap` for model explainability
  - `optuna` for hyperparameter optimization

- You may also want to set up a virtual environment:
  ```bash
  python -m venv mpnn_env
  source mpnn_env/bin/activate  # On Windows, use `mpnn_env\Scripts\activate`
  pip install -r requirements.txt
  ```

### Repository Structure
The key folders and files are structured as follows:

```
MPNN-MLP-Viscosity Prediction/
│
├── scripts/
│   ├── train_model.py                # Script to train the MPNN and MLP models
│   ├── optimize_hyperparameters.py   # Script for hyperparameter tuning using Optuna
│
├── src/
│   ├── data_processing/              # Data loading and processing scripts
│   ├── models/                       # Contains the MPNN, MLP, and traditional ML models
│   ├── evaluation/                   # Contains calculate metrics, and shap utils
│   ├── training/                     # Training and evaluation functions
│   ├── utils/                        # Utility functions like logging, feature extraction
│
├── requirements.txt                  # Dependencies
├── README.md                         # Overview of the project
├── setup.py                          # Setup script for package installation
└── LICENSE.txt
```

## Step-by-Step Tutorial
This section provides detailed instructions to help you run the entire pipeline, from preparing the data to training the model.

### Step 1: Data Preparation

#### 1.1 Obtain SMILES Strings
The input for our model can be obtained from the `MPNN-MLP-Viscosity prediction of DESs/src/data_processing/get_SMILES.py` document, which retrieves SMILES strings based on the English names of compounds.
- `SMILES1`: SMILES string for Component 1.
- `SMILES2`: SMILES string for Component 2.
- `X1`, `temperature`, `X2`, etc.: Additional numerical features for each compound.
- `Viscosity_cP`: Target column representing the viscosity value.

#### 1.2 Calculate Molecular Descriptors
The next step is to generate molecular descriptors from the SMILES strings. These descriptors, such as molecular weight and topological polar surface area (TPSA), provide additional information that can improve model predictions.

Use the script `src/data_processing/feature_engineering.py` to calculate the descriptors:

```python
from src.data_processing.feature_engineering import calculate_descriptors

# Example Usage
data = ['CCO', 'C1=CC=CC=C1']  # Replace with your SMILES list
descriptors_df = calculate_descriptors(data)
print(descriptors_df)
```
This script will take a list of SMILES strings and return a DataFrame containing the calculated descriptors.

### Step 2: Generate Graph Features

#### 2.1 Convert SMILES to Graph Representation
To capture the molecular structure more effectively, convert SMILES strings into graph representations using a Graph Neural Network (MPNN). Use the `smiles_to_molecule` function from `src/data_processing/smiles_utils.py`:

```python
from src.data_processing.smiles_utils import smiles_to_molecule

smiles = 'CCO'  # Example SMILES
graph_data = smiles_to_molecule(smiles)
print(graph_data)  # Graph data ready for GNN
```

### Step 3: Train the MPNN and MLP Models

Train the Message Passing Neural Network (MPNN) to extract meaningful graph-level features using the `train_model.py` script:

```bash
python scripts/train_model.py --data path/to/your/data.csv
```
This command will train the MPNN model using graph features generated from SMILES strings. After training, extract features for the Multi-Layer Perceptron (MLP) using the trained GNN model:

```python
from src.training.train_gnn import extract_features_and_cache

features_smiles1, _, _, _ = extract_features_and_cache(gnn_model, smiles_list1, device)
features_smiles2, _, _, _ = extract_features_and_cache(gnn_model, smiles_list2, device)
```
This function will return features that are passed to the MLP. Next, use the extracted graph features, combined with the numeric features, to train the MLP for predicting viscosity:

```bash
python scripts/train_model.py --data path/to/your/data.csv
```
This script trains the MLP using the combined feature set.

### Step 4: Hyperparameter Optimization
The model's performance can be further improved by optimizing hyperparameters such as learning rate, batch size, and the number of hidden layers. This can be achieved using `Optuna` for hyperparameter search.

Run the following script to start hyperparameter optimization:

```bash
python scripts/optimize_hyperparameters.py
```
The script will utilize Optuna to perform a search over possible hyperparameter configurations and return the best-performing set.

### Step 5: Evaluation and Model Explainability

#### 5.1 Model Evaluation
To evaluate the trained model, calculate metrics like Mean Squared Error (MSE), R-squared (R²), and Average Absolute Relative Deviation (AARD). Use the `calculate_metrics.py` script in `src/evaluation/`:

```python
from src.evaluation.calculate_metrics import calculate_metrics

# Example Usage
y_true = [3.0, 2.5, 4.0, 7.0]
y_pred = [2.8, 2.7, 3.9, 7.2]
metrics = calculate_metrics(y_true, y_pred)
print(metrics)
```
This will output a dictionary containing the calculated evaluation metrics.

#### 5.2 SHAP Analysis
To better understand the model's predictions, use SHAP (SHapley Additive exPlanations) for explainability:

```python
from src.utils.shap_utils import compute_shap_values

shap_values = compute_shap_values(smiles1, smiles2, numeric_features, mpnn_model, mlp_model, device)
print("SHAP values:", shap_values)
```
SHAP values provide insight into which features have the most significant impact on the prediction, which can be useful for debugging and understanding model behavior.

## Model Evaluation Script

The following script can be used to evaluate a pre-trained MLP model for DES viscosity prediction. It will load both the trained model and the data used during the training process, and output the performance metrics (MSE, R²) for both the training and test datasets.

### Usage
First, replace the file paths in the script (`model_path` and `data_path`) with the actual paths of your model and data files. (Shared file via cloud storage: Breadcrumbs MPNN-MLP for Viscosity Prediction of DESs, Link: https://pan.baidu.com/s/1of4EejchhFfQd0b9s4EJmg?pwd=43vd  Access code: 43vd)

Then, run the script: (Note:The saved model is from a single fold that performed well, so the prediction results may not match the performance of a model trained on the complete dataset.)

```bash
python evaluate_model.py
```

## Common Issues and Troubleshooting
- **ModuleNotFoundError**: Ensure that you have installed all dependencies correctly using `pip install -r requirements.txt`.
- **CUDA-related Issues**: Verify that your CUDA version is compatible with the installed PyTorch version. You can check compatibility [here](https://pytorch.org/get-started/previous-versions/).
- **RDKit Installation**: RDKit installation might be challenging on some systems. Refer to the official RDKit [installation guide](https://www.rdkit.org/docs/Install.html) for system-specific instructions.

## Future Work and Contributions
If you would like to contribute to this project, please feel free to fork the repository and submit a pull request. Potential areas of contribution include:
- **Improving Model Performance**: Experiment with additional molecular descriptors or advanced graph neural network architectures.
- **Feature Engineering**: Introduce new methods for feature extraction that could enhance the model's predictive power.
- **Adding New Models**: Integrate other machine learning models or improve the current GNN and MLP implementation.

