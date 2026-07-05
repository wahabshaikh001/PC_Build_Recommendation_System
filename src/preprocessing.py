import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def parse_capacity(capacity_str):
    if pd.isna(capacity_str):
        return 0.0
    if not isinstance(capacity_str, str):
        return float(capacity_str)
    cap = capacity_str.upper().strip()
    if 'TB' in cap:
        try:
            return float(cap.replace('TB', '').strip()) * 1000.0
        except:
            return 1000.0
    elif 'GB' in cap:
        try:
            return float(cap.replace('GB', '').strip())
        except:
            return 500.0
    try:
        return float(cap)
    except:
        return 0.0

def parse_speed(speed_str):
    if pd.isna(speed_str):
        return 0.0
    if not isinstance(speed_str, str):
        return float(speed_str)
    spd = speed_str.upper().strip()
    spd = spd.replace('MB/S', '').replace('MBPS', '').strip()
    try:
        return float(spd)
    except:
        return 0.0

def preprocess_datasets(datasets, models_dir="models"):
    """
    Cleans datasets, parses storage strings, handles missing values/duplicates,
    generates component names if missing, and fits & saves label encoders.
    """
    cleaned = {}
    
    # 1. CPU Preprocessing
    df_cpu = datasets['cpu'].copy()
    df_cpu.drop_duplicates(inplace=True)
    # Validate numeric columns
    numeric_cols = ['price', 'cores', 'threads', 'base_clock', 'boost_clock', 'tdp']
    for col in numeric_cols:
        df_cpu[col] = pd.to_numeric(df_cpu[col], errors='coerce')
        # If columns are completely empty, handle it
        median_val = df_cpu[col].median() if not df_cpu[col].isna().all() else 0.0
        df_cpu[col] = df_cpu[col].fillna(median_val)
        df_cpu.loc[df_cpu[col] <= 0, col] = median_val if median_val > 0 else 1.0
    df_cpu['integrated_graphics'] = df_cpu['integrated_graphics'].fillna(False).astype(bool)
    df_cpu['socket'] = df_cpu['socket'].fillna('Unknown').astype(str)
    cleaned['cpu'] = df_cpu

    # 2. GPU Preprocessing
    df_gpu = datasets['gpu'].copy()
    df_gpu.drop_duplicates(inplace=True)
    numeric_cols = ['price', 'vram', 'power_draw', 'benchmark_score']
    for col in numeric_cols:
        df_gpu[col] = pd.to_numeric(df_gpu[col], errors='coerce')
        median_val = df_gpu[col].median() if not df_gpu[col].isna().all() else 0.0
        df_gpu[col] = df_gpu[col].fillna(median_val)
        df_gpu.loc[df_gpu[col] <= 0, col] = median_val if median_val > 0 else 1.0
    cleaned['gpu'] = df_gpu

    # 3. RAM Preprocessing
    df_ram = datasets['ram'].copy()
    df_ram.drop_duplicates(inplace=True)
    numeric_cols = ['size', 'speed', 'price']
    for col in numeric_cols:
        df_ram[col] = pd.to_numeric(df_ram[col], errors='coerce')
        median_val = df_ram[col].median() if not df_ram[col].isna().all() else 0.0
        df_ram[col] = df_ram[col].fillna(median_val)
        df_ram.loc[df_ram[col] <= 0, col] = median_val if median_val > 0 else 1.0
    df_ram['type'] = df_ram['type'].fillna('Unknown').astype(str)
    # Generate name
    df_ram['name'] = df_ram.apply(lambda r: f"{r['type']} {int(r['size'])}GB {int(r['speed'])}MHz RAM", axis=1)
    cleaned['ram'] = df_ram

    # 4. Motherboard Preprocessing
    df_mobo = datasets['motherboard'].copy()
    df_mobo.drop_duplicates(inplace=True)
    numeric_cols = ['max_ram', 'price']
    for col in numeric_cols:
        df_mobo[col] = pd.to_numeric(df_mobo[col], errors='coerce')
        median_val = df_mobo[col].median() if not df_mobo[col].isna().all() else 0.0
        df_mobo[col] = df_mobo[col].fillna(median_val)
        df_mobo.loc[df_mobo[col] <= 0, col] = median_val if median_val > 0 else 1.0
    df_mobo['socket'] = df_mobo['socket'].fillna('Unknown').astype(str)
    df_mobo['chipset'] = df_mobo['chipset'].fillna('Unknown').astype(str)
    df_mobo['ram_type'] = df_mobo['ram_type'].fillna('Unknown').astype(str)
    cleaned['motherboard'] = df_mobo

    # 5. Storage Preprocessing
    df_storage = datasets['storage'].copy()
    df_storage.drop_duplicates(inplace=True)
    df_storage['capacity_gb'] = df_storage['capacity'].apply(parse_capacity)
    df_storage['speed_mbs'] = df_storage['speed'].apply(parse_speed)
    
    df_storage['price'] = pd.to_numeric(df_storage['price'], errors='coerce')
    median_price = df_storage['price'].median() if not df_storage['price'].isna().all() else 0.0
    df_storage['price'] = df_storage['price'].fillna(median_price)
    df_storage.loc[df_storage['price'] <= 0, 'price'] = median_price if median_price > 0 else 1.0
    
    # Generate name
    df_storage['name'] = df_storage.apply(lambda r: f"{r['capacity']} {r['type']} ({r['speed']})", axis=1)
    cleaned['storage'] = df_storage

    # 6. PSU Preprocessing
    df_psu = datasets['psu'].copy()
    df_psu.drop_duplicates(inplace=True)
    numeric_cols = ['wattage', 'price']
    for col in numeric_cols:
        df_psu[col] = pd.to_numeric(df_psu[col], errors='coerce')
        median_val = df_psu[col].median() if not df_psu[col].isna().all() else 0.0
        df_psu[col] = df_psu[col].fillna(median_val)
        df_psu.loc[df_psu[col] <= 0, col] = median_val if median_val > 0 else 1.0
    df_psu['efficiency_rating'] = df_psu['efficiency_rating'].fillna('Unknown').astype(str)
    df_psu['modular'] = df_psu['modular'].fillna('Unknown').astype(str)
    # Generate name
    df_psu['name'] = df_psu.apply(lambda r: f"{int(r['wattage'])}W {r['efficiency_rating']} PSU", axis=1)
    cleaned['psu'] = df_psu

    # Fit and save Label Encoders
    encoders = {}
    
    # socket encoder
    sockets = sorted(pd.concat([df_cpu['socket'], df_mobo['socket']]).unique().tolist()) + ['Unknown']
    le_socket = LabelEncoder()
    le_socket.fit(sockets)
    encoders['socket'] = le_socket
    
    # chipset encoder
    chipsets = sorted(df_mobo['chipset'].unique().tolist()) + ['Unknown']
    le_chipset = LabelEncoder()
    le_chipset.fit(chipsets)
    encoders['chipset'] = le_chipset
    
    # ram_type encoder
    ram_types = sorted(pd.concat([df_ram['type'], df_mobo['ram_type']]).unique().tolist()) + ['Unknown']
    le_ram_type = LabelEncoder()
    le_ram_type.fit(ram_types)
    encoders['ram_type'] = le_ram_type
    
    # efficiency_rating encoder
    ratings = sorted(df_psu['efficiency_rating'].unique().tolist()) + ['Unknown']
    le_rating = LabelEncoder()
    le_rating.fit(ratings)
    encoders['efficiency_rating'] = le_rating
    
    # storage type encoder
    storage_types = sorted(df_storage['type'].unique().tolist()) + ['Unknown']
    le_storage_type = LabelEncoder()
    le_storage_type.fit(storage_types)
    encoders['storage_type'] = le_storage_type
    
    os.makedirs(models_dir, exist_ok=True)
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    joblib.dump(encoders, encoders_path)
    
    # Apply encoders to datasets (as new columns)
    cleaned['cpu']['socket_encoded'] = le_socket.transform(cleaned['cpu']['socket'])
    cleaned['motherboard']['socket_encoded'] = le_socket.transform(cleaned['motherboard']['socket'])
    cleaned['motherboard']['chipset_encoded'] = le_chipset.transform(cleaned['motherboard']['chipset'])
    cleaned['ram']['ram_type_encoded'] = le_ram_type.transform(cleaned['ram']['type'])
    cleaned['motherboard']['ram_type_encoded'] = le_ram_type.transform(cleaned['motherboard']['ram_type'])
    cleaned['psu']['efficiency_encoded'] = le_rating.transform(cleaned['psu']['efficiency_rating'])
    cleaned['storage']['type_encoded'] = le_storage_type.transform(cleaned['storage']['type'])
    
    return cleaned
