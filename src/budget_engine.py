def get_target_budgets(user_budget):
    """
    Returns target budgets for Lower Grade, Medium Grade, and Best Grade.
    """
    return {
        'lower': 0.85 * user_budget,
        'medium': 1.00 * user_budget,
        'best': 1.10 * user_budget
    }

def get_component_budget_allocation(purpose):
    """
    Returns budget allocation fractions for each component based on purpose profile.
    Adjusts weights to ensure all 6 components receive a non-zero budget.
    """
    purpose_lower = purpose.lower().strip()
    if 'gaming' in purpose_lower:
        return {
            'cpu': 0.25,
            'gpu': 0.45,
            'ram': 0.10,
            'motherboard': 0.05,
            'storage': 0.05,
            'psu': 0.10
        }
    elif 'editing' in purpose_lower:
        # User weights: CPU 35%, RAM 25%, GPU 20%, Storage 15%, Motherboard 5%
        # PSU gets 8% budget, and other component budgets are scaled to sum to 100%
        return {
            'cpu': 0.32,
            'ram': 0.23,
            'gpu': 0.18,
            'storage': 0.14,
            'motherboard': 0.05,
            'psu': 0.08
        }
    else:  # Office
        # User weights: CPU 30%, RAM 25%, Storage 20%, Price Efficiency 25%
        # GPU, motherboard, and PSU are allocated 10% budget each
        return {
            'cpu': 0.30,
            'ram': 0.22,
            'storage': 0.18,
            'motherboard': 0.10,
            'psu': 0.10,
            'gpu': 0.10
        }
