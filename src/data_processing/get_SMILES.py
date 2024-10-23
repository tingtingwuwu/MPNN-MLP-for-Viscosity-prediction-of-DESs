import pandas as pd
import requests
import concurrent.futures
import time
from urllib.parse import quote

# Input and output file paths (update these paths as needed)
input_file_path = r' '  # Path to your input CSV file
output_file_path = r' '  # Path to save the output CSV file with SMILES data

# API base URLs for fetching SMILES strings
pubchem_base_url = ' '  # Base URL for PubChem API
chemspider_base_url = ' '  # Base URL for ChemSpider API (requires your API key)

# Cache dictionary to store previously queried compound SMILES to reduce redundant queries
cache = {}

# Load compound data from CSV file
df = pd.read_csv(input_file_path)

# Ensure the presence of Component#1 and Component#2 columns in the dataset
compound_names_1 = df['Component#1']  # Extract Component#1 column
compound_names_2 = df['Component#2']  # Extract Component#2 column


# Function to get SMILES from PubChem
def get_pubchem_smiles(name, retry_count=3, delay=1):
    """
    Get SMILES string for a compound using PubChem API.

    Parameters:
    name (str): Compound name.
    retry_count (int): Number of retry attempts if the request fails.
    delay (int): Delay (in seconds) between retries.

    Returns:
    str: SMILES string for the compound, or 'Not Found' if unsuccessful.
    """
    if name in cache:
        return cache[name]  # Use cached value if available

    name_encoded = quote(name)  # URL encode the compound name

    for attempt in range(retry_count):
        try:
            # Make a request to PubChem API for the compound's SMILES
            response = requests.get(pubchem_base_url.format(name_encoded), timeout=5)
            if response.status_code == 200:
                # Extract SMILES from the response (assuming a specific CSV format)
                smiles = response.text.split('\n')[1].split(',')[1]
                cache[name] = smiles  # Store result in cache for future use
                return smiles
            else:
                # Log error message if API response status is not successful
                print(f"Error fetching SMILES for {name}: HTTP Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            # Log exception message and retry after a delay
            print(f"Error fetching SMILES for {name}: {e}")
            time.sleep(delay)

    return 'Not Found'  # Return 'Not Found' if all retries fail


# Function to get SMILES from ChemSpider as a backup option
def get_chemspider_smiles(name):
    """
    Get SMILES string for a compound using ChemSpider API.

    Parameters:
    name (str): Compound name.

    Returns:
    str: SMILES string for the compound, or 'Not Found' if unsuccessful.
    """
    try:
        # Make a request to ChemSpider API
        response = requests.get(chemspider_base_url.format(quote(name)))
        if response.status_code == 200:
            # Process the response to extract the SMILES value (custom parsing required)
            smiles = 'ChemSpider_Smiles_Placeholder'  # Replace this with correct parsing code
            cache[name] = smiles
            return smiles
        else:
            print(f"Error fetching SMILES for {name} from ChemSpider: HTTP Status {response.status_code}")
    except Exception as e:
        print(f"Error fetching SMILES for {name} from ChemSpider: {e}")

    return 'Not Found'  # Return 'Not Found' if request fails


# Function to get SMILES using PubChem first, then ChemSpider if needed
def get_smiles(name):
    """
    Retrieve SMILES string for a compound, first trying PubChem and then ChemSpider as a fallback.

    Parameters:
    name (str): Compound name.

    Returns:
    str: SMILES string for the compound, or 'Not Found' if neither source succeeds.
    """
    smiles = get_pubchem_smiles(name)
    if smiles == 'Not Found':
        print(f"Trying ChemSpider for {name}...")
        smiles = get_chemspider_smiles(name)

    return smiles


# Use parallel processing to fetch SMILES strings for multiple compounds concurrently
def fetch_smiles(names):
    """
    Fetch SMILES strings for a list of compound names using parallel processing.

    Parameters:
    names (list of str): List of compound names.

    Returns:
    list of str: List of SMILES strings.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        smiles = list(executor.map(get_smiles, names))
    return smiles


# Fetch SMILES strings for Component#1 and Component#2
smiles_list_1 = fetch_smiles(compound_names_1)
smiles_list_2 = fetch_smiles(compound_names_2)

# Add SMILES strings as new columns in the dataframe
df['Component#1_SMILES'] = smiles_list_1
df['Component#2_SMILES'] = smiles_list_2

# Save updated dataframe with SMILES information to a new CSV file
df.to_csv(output_file_path, index=False)

print("Processing complete, SMILES information has been written to the output file.")
