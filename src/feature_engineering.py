import pandas as pd

def compute_cpu_score(df):
    return (df['cores'] * 5) + (df['threads'] * 3) + (df['base_clock'] * 10) + (df['boost_clock'] * 15)

def compute_gpu_score(df):
    return df['benchmark_score']

def compute_ram_score(df):
    return df['size'] * df['speed']

def compute_storage_score(df):
    return df['capacity_gb'] * df['speed_mbs']

def compute_psu_score(df):
    # Bronze = 1, Silver = 2, Gold = 3, Platinum = 4, Titanium = 5
    rating_map = {
        'BRONZE': 1.0,
        'SILVER': 2.0,
        'GOLD': 3.0,
        'PLATINUM': 4.0,
        'TITANIUM': 5.0,
        '80+ BRONZE': 1.0,
        '80+ SILVER': 2.0,
        '80+ GOLD': 3.0,
        '80+ PLATINUM': 4.0,
        '80+ TITANIUM': 5.0
    }
    
    def map_rating(rating):
        if not isinstance(rating, str):
            return 0.0
        r_upper = rating.upper().strip()
        for key, score in rating_map.items():
            if key in r_upper:
                return score
        # For non-certified or white, return a default low score (e.g. 0.5)
        if 'WHITE' in r_upper or 'STANDARD' in r_upper:
            return 0.5
        return 0.0
        
    return df['efficiency_rating'].apply(map_rating)

def add_scores_to_datasets(datasets):
    """
    Applies custom component scores to each dataset and adds score columns.
    """
    datasets['cpu']['cpu_score'] = compute_cpu_score(datasets['cpu'])
    datasets['gpu']['gpu_score'] = compute_gpu_score(datasets['gpu'])
    datasets['ram']['ram_score'] = compute_ram_score(datasets['ram'])
    datasets['storage']['storage_score'] = compute_storage_score(datasets['storage'])
    datasets['psu']['psu_score'] = compute_psu_score(datasets['psu'])
    # Motherboard performance score can be max_ram
    datasets['motherboard']['motherboard_score'] = datasets['motherboard']['max_ram'].astype(float)
    return datasets
