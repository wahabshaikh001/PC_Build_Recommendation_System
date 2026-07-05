def get_performance_weights(purpose):
    purpose_lower = purpose.lower().strip()
    if 'gaming' in purpose_lower:
        return {
            'cpu': 0.25,
            'gpu': 0.45,
            'ram': 0.10,
            'storage': 0.05,
            'motherboard': 0.05,
            'psu': 0.10
        }
    elif 'editing' in purpose_lower:
        return {
            'cpu': 0.35,
            'ram': 0.25,
            'gpu': 0.20,
            'storage': 0.15,
            'motherboard': 0.05,
            'psu': 0.00
        }
    else:  # Office
        return {
            'cpu': 0.30,
            'ram': 0.25,
            'storage': 0.20,
            'price_efficiency': 0.25
        }

def calculate_build_performance(cpu, gpu, ram, motherboard, storage, psu, purpose, max_scores, total_cost=None, user_budget=None):
    """
    Calculates the composite performance score (0 to 100) for a combination of components.
    """
    # Normalize component scores
    cpu_score_norm = cpu['cpu_score'] / max_scores.get('cpu', 1.0)
    gpu_score_norm = gpu['gpu_score'] / max_scores.get('gpu', 1.0)
    ram_score_norm = ram['ram_score'] / max_scores.get('ram', 1.0)
    storage_score_norm = storage['storage_score'] / max_scores.get('storage', 1.0)
    psu_score_norm = psu['psu_score'] / max_scores.get('psu', 1.0)
    
    # Motherboard max_ram normalized
    mobo_score_norm = motherboard['max_ram'] / max_scores.get('motherboard', 128.0)
    
    weights = get_performance_weights(purpose)
    performance = 0.0
    purpose_lower = purpose.lower().strip()
    
    if 'gaming' in purpose_lower:
        performance += weights['cpu'] * cpu_score_norm
        performance += weights['gpu'] * gpu_score_norm
        performance += weights['ram'] * ram_score_norm
        performance += weights['storage'] * storage_score_norm
        performance += weights['motherboard'] * mobo_score_norm
        performance += weights['psu'] * psu_score_norm
    elif 'editing' in purpose_lower:
        performance += weights['cpu'] * cpu_score_norm
        performance += weights['ram'] * ram_score_norm
        performance += weights['gpu'] * gpu_score_norm
        performance += weights['storage'] * storage_score_norm
        performance += weights['motherboard'] * mobo_score_norm
    else:  # Office
        performance += weights['cpu'] * cpu_score_norm
        performance += weights['ram'] * ram_score_norm
        performance += weights['storage'] * storage_score_norm
        
        # Calculate price efficiency score (0 to 1)
        if total_cost is not None and user_budget is not None:
            # Rewards lower costs relative to the budget
            price_efficiency = max(0.0, 1.0 - (total_cost / user_budget))
        else:
            price_efficiency = 0.5
            
        performance += weights['price_efficiency'] * price_efficiency
        
    return performance * 100.0
