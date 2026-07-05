from src.performance_engine import calculate_build_performance

def get_final_recommendation(builds_dict, user_budget, purpose, max_scores):
    """
    Evaluates Lower, Medium, and Best Grade builds using the final AI Build Score Engine.
    Applies the budget penalty system and recommends the best build.
    Returns (recommended_category, updated_builds, confidence, reason).
    """
    updated_builds = {}
    
    for grade, build in builds_dict.items():
        if build is None:
            continue
            
        # 1. Performance Score (recalculated relative to user_budget)
        perf_score = calculate_build_performance(
            build['cpu'], build['gpu'], build['ram'], build['motherboard'], 
            build['storage'], build['psu'], purpose, max_scores, 
            build['total_cost'], user_budget
        )
        
        # 2. Budget Efficiency (relative to user_budget)
        budget_eff = max(0.0, 100.0 * (1.0 - abs(build['total_cost'] - user_budget) / user_budget))
        
        # 3. Purpose Matching Score (from build generation similarity)
        purpose_match = build['purpose_match_score']
        
        # 4. Compatibility Score
        compatibility = 100.0
        
        # Recalculate Build Score
        raw_build_score = (0.40 * perf_score + 
                           0.30 * budget_eff + 
                           0.20 * purpose_match + 
                           0.10 * compatibility)
        
        # 5. Budget Penalty System
        penalty = 0.0
        if build['total_cost'] > user_budget:
            # Over-budget penalty: linear + quadratic component
            ratio_over = (build['total_cost'] - user_budget) / user_budget
            penalty = 50.0 * ratio_over + 100.0 * (ratio_over ** 2)
            
        final_build_score = max(0.0, raw_build_score - penalty)
        
        # Update build dictionary
        build_copy = build.copy()
        build_copy['performance_score'] = perf_score
        build_copy['budget_efficiency'] = budget_eff
        build_copy['purpose_match_score'] = purpose_match
        build_copy['compatibility_score'] = compatibility
        build_copy['penalty'] = penalty
        build_copy['build_score'] = final_build_score
        
        updated_builds[grade] = build_copy
        
    if not updated_builds:
        return None, {}, 0.0, "No compatible builds could be generated for your budget and purpose."
        
    # Find the build with the highest final build score
    best_grade = None
    best_score = -1.0
    for grade, build in updated_builds.items():
        if build['build_score'] > best_score:
            best_score = build['build_score']
            best_grade = grade
            
    recommended_build = updated_builds[best_grade]
    
    # Calculate confidence score based on the recommended build's Build Score
    confidence = min(99.0, max(50.0, recommended_build['build_score']))
    
    # Generate reason
    cost = recommended_build['total_cost']
    perf = recommended_build['performance_score']
    
    reasons = []
    if best_grade == 'lower':
        reasons.append(f"it saves you ${user_budget - cost:.2f} (costing only ${cost:.2f} against your ${user_budget:.2f} budget)")
        reasons.append("while maintaining solid compatibility and highly satisfactory performance.")
    elif best_grade == 'medium':
        reasons.append(f"it utilizes your budget optimally at ${cost:.2f} (almost exactly matching your ${user_budget:.2f} budget)")
        reasons.append("providing the most balanced price-to-performance ratio for your needs.")
    else:  # best grade
        reasons.append(f"although it slightly exceeds your budget by ${cost - user_budget:.2f} (costing ${cost:.2f}),")
        reasons.append(f"the significant performance leap ({perf:.1f} vs other configurations) makes the slight budget stretch highly worthwhile.")
        
    reason_str = f"The {best_grade.upper()} GRADE build is recommended because " + " ".join(reasons)
    
    return best_grade, updated_builds, confidence, reason_str
