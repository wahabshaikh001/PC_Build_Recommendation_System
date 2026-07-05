import os
import pandas as pd

def load_datasets(dataset_dir=None):
    """
    Loads CPU, GPU, RAM, Motherboard, Storage, and PSU datasets from the dataset directory.
    Uses case-insensitive substring matching to locate files for each component,
    prioritizing .csv format and falling back to .xlsx if necessary.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DATASET")
        
    datasets = {}
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        
    files = os.listdir(dataset_dir)
    
    # Define keywords to match files dynamically
    keywords = {
        'cpu': ['cpu'],
        'gpu': ['gpu'],
        'ram': ['ram'],
        'motherboard': ['motherboard', 'mobo'],
        'storage': ['storage', 'ssd', 'hdd'],
        'psu': ['psu', 'power']
    }
    
    for component, keys in keywords.items():
        matched_file = None
        
        # Phase 1: Try to match with .csv files first
        for filename in files:
            fn_lower = filename.lower()
            if filename.endswith('.csv') and any(k in fn_lower for k in keys):
                matched_file = filename
                break
                
        # Phase 2: If no csv matches, try to match with .xlsx files
        if not matched_file:
            for filename in files:
                fn_lower = filename.lower()
                if filename.endswith('.xlsx') and any(k in fn_lower for k in keys):
                    matched_file = filename
                    break
                    
        # Phase 3: Fallback to any file containing keywords
        if not matched_file:
            for filename in files:
                fn_lower = filename.lower()
                if any(k in fn_lower for k in keys):
                    matched_file = filename
                    break
                    
        if not matched_file:
            raise FileNotFoundError(f"Could not dynamically identify a dataset file for '{component}' in {dataset_dir}. Searched keys: {keys}")
            
        file_path = os.path.join(dataset_dir, matched_file)
        
        try:
            if matched_file.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            datasets[component] = df
        except Exception as e:
            raise IOError(f"Failed to read dataset file '{file_path}': {e}")
            
    # Normalize RAM dataset columns: rename size_gb -> size, speed_mhz -> speed if present
    if 'ram' in datasets:
        ram_df = datasets['ram']
        rename_map = {}
        if 'size_gb' in ram_df.columns:
            rename_map['size_gb'] = 'size'
        if 'speed_mhz' in ram_df.columns:
            rename_map['speed_mhz'] = 'speed'
        if rename_map:
            datasets['ram'] = ram_df.rename(columns=rename_map)
            
    return datasets
