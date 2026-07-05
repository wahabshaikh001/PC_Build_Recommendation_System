import pandas as pd
import numpy as np
from src.compatibility_engine import check_compatibility
from src.budget_engine import get_component_budget_allocation
from src.similarity_engine import rank_components
from src.performance_engine import calculate_build_performance

def generate_builds_for_target(datasets, purpose, target_budget, max_scores, user_budget, top_n=15):
    """
    Generates a list of compatible builds for a target budget and ranks them by Build Score.
    Employs a compatibility-first search to ensure that compatible component matches are
    always found. Uses highly optimized search widths for performance.
    """
    # 1. Get component budget allocations
    allocations = get_component_budget_allocation(purpose)
    
    # 2. Rank components using Cosine Similarity against component target prices
    ranked_cpu = rank_components(datasets['cpu'], 'cpu', allocations['cpu'] * target_budget, max_scores, target_budget)
    ranked_gpu = rank_components(datasets['gpu'], 'gpu', allocations['gpu'] * target_budget, max_scores, target_budget)
    ranked_ram = rank_components(datasets['ram'], 'ram', allocations['ram'] * target_budget, max_scores, target_budget)
    ranked_mobo = rank_components(datasets['motherboard'], 'motherboard', allocations['motherboard'] * target_budget, max_scores, target_budget)
    ranked_storage = rank_components(datasets['storage'], 'storage', allocations['storage'] * target_budget, max_scores, target_budget)
    ranked_psu = rank_components(datasets['psu'], 'psu', allocations['psu'] * target_budget, max_scores, target_budget)
    
    # Convert dataframes to dictionaries for rapid iteration
    cpus = ranked_cpu.to_dict('records')
    gpus = ranked_gpu.to_dict('records')
    rams = ranked_ram.to_dict('records')
    mobos = ranked_mobo.to_dict('records')
    storages = ranked_storage.to_dict('records')
    psus = ranked_psu.to_dict('records')
    
    # Search widths: select primary drivers globally (optimized values)
    search_cpus = cpus[:top_n]
    search_gpus = gpus[:top_n]
    search_storages = storages[:5] # top 5 storages
    
    best_build = None
    best_score = -1.0
    
    # 3. Dynamic backtracking search with early pruning
    for cpu in search_cpus:
        # Find compatible Motherboards from the entire dataset, keeping the top 5 by similarity
        comp_mobos = [m for m in mobos if m['socket'] == cpu['socket']][:5]
        for mobo in comp_mobos:
            # Find RAMs matching the Motherboard's generation, keeping the top 5 by similarity
            comp_rams = [r for r in rams if r['type'] == mobo['ram_type']][:5]
            for ram in comp_rams:
                for gpu in search_gpus:
                    # Find PSUs matching the wattage requirements, keeping the top 3 by similarity
                    min_wattage = (cpu['tdp'] + gpu['power_draw']) * 1.30
                    comp_psus = [p for p in psus if p['wattage'] >= min_wattage][:3]
                    for psu in comp_psus:
                        for storage in search_storages:
                            # Calculate total cost
                            total_cost = (cpu['price'] + gpu['price'] + ram['price'] + 
                                          mobo['price'] + storage['price'] + psu['price'])
                            
                            # Budget constraint: allow up to 1.35 times the target budget
                            if total_cost > target_budget * 1.35:
                                continue
                                
                            # Calculate Build Score components
                            avg_similarity = (cpu['similarity'] + gpu['similarity'] + ram['similarity'] + 
                                              mobo['similarity'] + storage['similarity'] + psu['similarity']) / 6.0
                            
                            perf_score = calculate_build_performance(
                                cpu, gpu, ram, mobo, storage, psu, purpose, max_scores, total_cost, user_budget
                            )
                            
                            budget_efficiency = max(0.0, 100.0 * (1.0 - abs(total_cost - target_budget) / target_budget))
                            
                            purpose_match_score = avg_similarity * 100.0
                            compatibility_score = 100.0
                            
                            # Calculate final Build Score
                            build_score = (0.40 * perf_score + 
                                           0.30 * budget_efficiency + 
                                           0.20 * purpose_match_score + 
                                           0.10 * compatibility_score)
                            
                            if build_score > best_score:
                                best_score = build_score
                                best_build = {
                                    'cpu': cpu,
                                    'gpu': gpu,
                                    'ram': ram,
                                    'motherboard': mobo,
                                    'storage': storage,
                                    'psu': psu,
                                    'total_cost': total_cost,
                                    'total_power': cpu['tdp'] + gpu['power_draw'],
                                    'cpu_score': cpu['cpu_score'],
                                    'gpu_score': gpu['gpu_score'],
                                    'ram_score': ram['ram_score'],
                                    'storage_score': storage['storage_score'],
                                    'psu_score': psu['psu_score'],
                                    'avg_similarity': avg_similarity,
                                    'performance_score': perf_score,
                                    'budget_efficiency': budget_efficiency,
                                    'purpose_match_score': purpose_match_score,
                                    'compatibility_score': compatibility_score,
                                    'build_score': build_score
                                }
                                
    # Fallback search if no builds are found (specifically for extremely low budgets)
    # Relax the budget limit filter to 2.0x target_budget
    if best_build is None:
        for cpu in search_cpus:
            comp_mobos = [m for m in mobos if m['socket'] == cpu['socket']][:5]
            for mobo in comp_mobos:
                comp_rams = [r for r in rams if r['type'] == mobo['ram_type']][:5]
                for ram in comp_rams:
                    for gpu in search_gpus:
                        min_wattage = (cpu['tdp'] + gpu['power_draw']) * 1.30
                        comp_psus = [p for p in psus if p['wattage'] >= min_wattage][:3]
                        for psu in comp_psus:
                            for storage in search_storages:
                                total_cost = (cpu['price'] + gpu['price'] + ram['price'] + 
                                              mobo['price'] + storage['price'] + psu['price'])
                                
                                # Relaxed budget constraint
                                if total_cost > target_budget * 2.0:
                                    continue
                                    
                                avg_similarity = (cpu['similarity'] + gpu['similarity'] + ram['similarity'] + 
                                                  mobo['similarity'] + storage['similarity'] + psu['similarity']) / 6.0
                                
                                perf_score = calculate_build_performance(
                                    cpu, gpu, ram, mobo, storage, psu, purpose, max_scores, total_cost, user_budget
                                )
                                
                                budget_efficiency = max(0.0, 100.0 * (1.0 - abs(total_cost - target_budget) / target_budget))
                                purpose_match_score = avg_similarity * 100.0
                                compatibility_score = 100.0
                                
                                build_score = (0.40 * perf_score + 
                                               0.30 * budget_efficiency + 
                                               0.20 * purpose_match_score + 
                                               0.10 * compatibility_score)
                                
                                if build_score > best_score:
                                    best_score = build_score
                                    best_build = {
                                        'cpu': cpu,
                                        'gpu': gpu,
                                        'ram': ram,
                                        'motherboard': mobo,
                                        'storage': storage,
                                        'psu': psu,
                                        'total_cost': total_cost,
                                        'total_power': cpu['tdp'] + gpu['power_draw'],
                                        'cpu_score': cpu['cpu_score'],
                                        'gpu_score': gpu['gpu_score'],
                                        'ram_score': ram['ram_score'],
                                        'storage_score': storage['storage_score'],
                                        'psu_score': psu['psu_score'],
                                        'avg_similarity': avg_similarity,
                                        'performance_score': perf_score,
                                        'budget_efficiency': budget_efficiency,
                                        'purpose_match_score': purpose_match_score,
                                        'compatibility_score': compatibility_score,
                                        'build_score': build_score
                                    }
                                    
    return best_build
