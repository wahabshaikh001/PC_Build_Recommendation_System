def check_compatibility(cpu, gpu, ram, motherboard, psu):
    """
    Validates compatibility for a build combination.
    Returns (is_compatible, list_of_failure_reasons).
    """
    reasons = []
    
    # 1. CPU and Motherboard socket match
    if cpu['socket'] != motherboard['socket']:
        reasons.append(f"CPU socket ({cpu['socket']}) does not match Motherboard socket ({motherboard['socket']})")
        
    # 2. RAM and Motherboard RAM type match
    if ram['type'] != motherboard['ram_type']:
        reasons.append(f"RAM type ({ram['type']}) does not match Motherboard RAM type ({motherboard['ram_type']})")
        
    # 3. PSU Wattage Validation
    required_wattage = (cpu['tdp'] + gpu['power_draw']) * 1.30
    if psu['wattage'] < required_wattage:
        reasons.append(f"PSU wattage ({psu['wattage']}W) is insufficient. Required: {required_wattage:.1f}W (CPU TDP {cpu['tdp']}W + GPU {gpu['power_draw']}W + 30% overhead)")
        
    return len(reasons) == 0, reasons
