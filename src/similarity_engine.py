import numpy as np
import pandas as pd

def filter_by_budget_limit(df, component_type, target_budget):
    """
    Filters components to exclude those that are too expensive for the total target budget.
    """
    if df.empty:
        return df
        
    # Maximum price fraction of the target budget allowed for each component type
    fraction_map = {
        'cpu': 0.48,
        'gpu': 0.58,
        'ram': 0.22,
        'motherboard': 0.28,
        'storage': 0.22,
        'psu': 0.22
    }
    
    fraction = fraction_map.get(component_type, 0.25)
    
    # If the target budget is very small, we must relax the fraction to allow base components
    if target_budget < 450:
        fraction = max(fraction, 0.60 if component_type in ['cpu', 'gpu'] else 0.40)
        
    limit = target_budget * fraction
    
    # Ensure the limit is at least slightly higher than the minimum price in the dataset
    min_price = df['price'].min()
    limit = max(limit, min_price + 10.0)
    
    return df[df['price'] <= limit]

def compute_cosine_similarity(df, feature_cols, ideal_vector):
    """
    Computes cosine similarity between each row in df (for feature_cols) and the ideal_vector.
    Normalizes columns locally to [0, 1] range first to prevent scale dominance.
    """
    if df.empty:
        return np.array([])
        
    X = df[feature_cols].values.astype(float)
    
    # Calculate min and max for each column
    min_vals = df[feature_cols].min().values.astype(float)
    max_vals = df[feature_cols].max().values.astype(float)
    
    # Handle zero range
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    # Scale features and ideal vector to [0, 1]
    X_scaled = (X - min_vals) / range_vals
    ideal_scaled = (np.array(ideal_vector).astype(float) - min_vals) / range_vals
    
    # Compute cosine similarity
    dot_products = np.dot(X_scaled, ideal_scaled)
    norms_X = np.linalg.norm(X_scaled, axis=1)
    norm_ideal = np.linalg.norm(ideal_scaled)
    
    # Avoid division by zero
    norms_X[norms_X == 0] = 1e-9
    if norm_ideal == 0:
        norm_ideal = 1e-9
        
    similarities = dot_products / (norms_X * norm_ideal)
    return np.clip(similarities, 0.0, 1.0)

def get_ideal_vectors_and_features(component_type, target_price, df, max_scores):
    """
    Returns the feature columns and the ideal vector for a component type.
    """
    if component_type == 'cpu':
        feature_cols = ['price', 'cpu_score', 'cores', 'threads', 'base_clock', 'boost_clock', 'tdp']
        min_tdp = df['tdp'].min() if not df.empty else 35.0
        ideal = [target_price, max_scores.get('cpu', 1.0), df['cores'].max(), df['threads'].max(), 
                 df['base_clock'].max(), df['boost_clock'].max(), min_tdp]
    elif component_type == 'gpu':
        feature_cols = ['price', 'gpu_score', 'vram', 'power_draw']
        min_power = df['power_draw'].min() if not df.empty else 50.0
        ideal = [target_price, max_scores.get('gpu', 1.0), df['vram'].max(), min_power]
    elif component_type == 'ram':
        feature_cols = ['price', 'ram_score', 'size', 'speed']
        ideal = [target_price, max_scores.get('ram', 1.0), df['size'].max(), df['speed'].max()]
    elif component_type == 'motherboard':
        feature_cols = ['price', 'motherboard_score', 'max_ram']
        ideal = [target_price, max_scores.get('motherboard', 1.0), df['max_ram'].max()]
    elif component_type == 'storage':
        feature_cols = ['price', 'storage_score', 'capacity_gb', 'speed_mbs']
        ideal = [target_price, max_scores.get('storage', 1.0), df['capacity_gb'].max(), df['speed_mbs'].max()]
    elif component_type == 'psu':
        feature_cols = ['price', 'psu_score', 'wattage']
        ideal = [target_price, max_scores.get('psu', 1.0), df['wattage'].max()]
    else:
        raise ValueError(f"Unknown component type: {component_type}")
        
    return feature_cols, ideal

def rank_components(df, component_type, target_price, max_scores, target_budget):
    """
    Computes cosine similarity for all components in the dataframe and adds it as a column.
    Filters out components that exceed budget limits first.
    Returns the dataframe sorted by similarity descending.
    """
    if df.empty:
        return df.copy()
        
    df_filtered = filter_by_budget_limit(df, component_type, target_budget)
    
    if df_filtered.empty:
        df_filtered = df.copy()
        
    feature_cols, ideal = get_ideal_vectors_and_features(component_type, target_price, df_filtered, max_scores)
    
    similarities = compute_cosine_similarity(df_filtered, feature_cols, ideal)
    df_copy = df_filtered.copy()
    df_copy['similarity'] = similarities
    
    return df_copy.sort_values(by='similarity', ascending=False)
