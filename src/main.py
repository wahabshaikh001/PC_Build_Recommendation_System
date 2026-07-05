import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the import path so that 'src' can be imported when running as a direct script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_datasets
from src.preprocessing import preprocess_datasets
from src.feature_engineering import add_scores_to_datasets
from src.build_generator import generate_builds_for_target
from src.recommendation_engine import get_final_recommendation

def display_build(grade, build, user_budget, predicted_quality):
    """
    Prints a formatted summary of a single build grade.
    """
    if build is None:
        print(f"\n==================== {grade.upper()} GRADE BUILD ====================")
        print("No compatible build could be generated for this budget class.")
        print("=================================================================\n")
        return
        
    print(f"\n==================== {grade.upper()} GRADE BUILD ====================")
    print(f"{'Component':<12} | {'Name':<50} | {'Price':<10}")
    print("-" * 78)
    print(f"{'CPU':<12} | {build['cpu']['name']:<50} | ${build['cpu']['price']:<9.2f}")
    print(f"{'GPU':<12} | {build['gpu']['name']:<50} | ${build['gpu']['price']:<9.2f}")
    print(f"{'RAM':<12} | {build['ram']['name']:<50} | ${build['ram']['price']:<9.2f}")
    print(f"{'Motherboard':<12} | {build['motherboard']['name']:<50} | ${build['motherboard']['price']:<9.2f}")
    print(f"{'Storage':<12} | {build['storage']['name']:<50} | ${build['storage']['price']:<9.2f}")
    print(f"{'PSU':<12} | {build['psu']['name']:<50} | ${build['psu']['price']:<9.2f}")
    psu_modular = build['psu'].get('modular', 'Unknown')
    modularity = "Fully Modular" if psu_modular == "Yes" else "Non-Modular" if psu_modular == "No" else psu_modular
    print(f"{'PSU Mod':<12} | {modularity:<50} | {'':<9}")
    print("-" * 78)
    print(f"Total Cost:              ${build['total_cost']:.2f}")
    print(f"User Budget:             ${user_budget:.2f}")
    print(f"Budget Difference:       ${build['total_cost'] - user_budget:+.2f}")
    print(f"Performance Score:       {build['performance_score']:.1f} / 100")
    print(f"Compatibility Status:    COMPATIBLE")
    print(f"Predicted Build Quality: {predicted_quality:.1f} / 100")
    print("=================================================================\n")

def generate_run_graphs(builds, user_budget):
    """
    Generates run-specific graphs (Build Cost Comparison and Build Performance Comparison).
    """
    os.makedirs("results/graphs", exist_ok=True)
    
    grades = []
    costs = []
    performances = []
    
    for grade in ['lower', 'medium', 'best']:
        build = builds.get(grade)
        if build is not None:
            grades.append(grade.upper())
            costs.append(build['total_cost'])
            performances.append(build['performance_score'])
            
    if not grades:
        return
        
    # Chart 6: Build Cost Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(grades, costs, color=['#F8B195', '#F67280', '#C06C84'], edgecolor='black', width=0.5)
    plt.axhline(y=user_budget, color='r', linestyle='--', linewidth=1.5, label=f'User Budget (${user_budget:.0f})')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 10, f"${height:.2f}", ha='center', va='bottom', fontweight='bold')
        
    plt.ylabel('Cost (USD)')
    plt.title('Build Cost Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/graphs/build_cost_comparison.png', dpi=150)
    plt.close()
    
    # Chart 7: Build Performance Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(grades, performances, color=['#99B898', '#FECEAB', '#FF847C'], edgecolor='black', width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.1f}", ha='center', va='bottom', fontweight='bold')
        
    plt.ylabel('Performance Score (0-100)')
    plt.title('Build Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/graphs/build_performance_comparison.png', dpi=150)
    plt.close()
    print("Run-specific comparison graphs saved to results/graphs/ folder.")

def main():
    print("=" * 60)
    print("      AI-BASED PC BUILD RECOMMENDATION SYSTEM (CLI)      ")
    print("=" * 60)
    
    # Determine project root path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if models exist
    required_files = [
        os.path.join(project_root, "models", "linear_regression.pkl"),
        os.path.join(project_root, "models", "random_forest.pkl"),
        os.path.join(project_root, "models", "xgboost.pkl"),
        os.path.join(project_root, "models", "scaler.pkl"),
        os.path.join(project_root, "models", "label_encoders.pkl"),
        os.path.join(project_root, "models", "max_scores.pkl")
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            print(f"Error: Required model file {rf} is missing.")
            print("Please run train_models.py followed by evaluate_models.py to prepare models.")
            return

    # User Input
    try:
        user_budget = float(input("\nEnter Budget (USD): $"))
        if user_budget <= 0:
            print("Budget must be a positive number.")
            return
    except ValueError:
        print("Invalid budget input. Please enter a numerical value.")
        return
        
    print("\nSelect Purpose:")
    print("1. Gaming")
    print("2. Editing")
    print("3. Office")
    choice = input("Enter choice (1-3): ").strip()
    
    purpose_map = {'1': 'Gaming', '2': 'Editing', '3': 'Office'}
    if choice not in purpose_map:
        print("Invalid choice. Selecting 'Gaming' by default.")
        purpose = 'Gaming'
    else:
        purpose = purpose_map[choice]
        
    print(f"\nUser Input Received: Budget = ${user_budget:.2f}, Purpose = {purpose}")
    print("Analyzing components and searching for optimal configurations...")
    
    # 1. Load component datasets
    datasets = load_datasets()
    datasets = preprocess_datasets(datasets)
    datasets = add_scores_to_datasets(datasets)
    
    # Load scaling and models
    max_scores = joblib.load(os.path.join(project_root, "models", "max_scores.pkl"))
    scaler = joblib.load(os.path.join(project_root, "models", "scaler.pkl"))
    model_lr = joblib.load(os.path.join(project_root, "models", "linear_regression.pkl"))
    model_rf = joblib.load(os.path.join(project_root, "models", "random_forest.pkl"))
    model_xgb = joblib.load(os.path.join(project_root, "models", "xgboost.pkl"))
    model_best = joblib.load(os.path.join(project_root, "models", "best_model.pkl"))
    best_model_info = joblib.load(os.path.join(project_root, "models", "best_model_info.pkl"))
    best_model_name = best_model_info.get('best_model_name', 'XGBoost Regressor')
    
    # 2. Get budgets for Lower, Medium, Best Grades
    target_budgets = {
        'lower': 0.85 * user_budget,
        'medium': 1.00 * user_budget,
        'best': 1.10 * user_budget
    }
    
    # 3. Generate optimal builds
    raw_builds = {}
    for grade, target in target_budgets.items():
        # Using top_n=20 for quick but comprehensive search
        raw_builds[grade] = generate_builds_for_target(datasets, purpose, target, max_scores, user_budget, top_n=20)
        
    # 4. Final scoring and penalty system
    recommended_grade, final_builds, confidence, reason = get_final_recommendation(
        raw_builds, user_budget, purpose, max_scores
    )
    
    # 5. Predict quality using ML models
    predicted_qualities = {}
    purpose_encoded_map = {'Gaming': 0, 'Editing': 1, 'Office': 2}
    
    for grade, build in final_builds.items():
        if build is None:
            continue
            
        # Structure features for model input
        features_df = pd.DataFrame([{
            'total_cost': build['total_cost'],
            'total_power': build['total_power'],
            'cpu_score': build['cpu_score'],
            'gpu_score': build['gpu_score'],
            'ram_score': build['ram_score'],
            'storage_score': build['storage_score'],
            'psu_score': build['psu_score'],
            'purpose_encoded': purpose_encoded_map[purpose],
            'compatibility': 1.0,  # compatible by generation
            'budget_difference': build['total_cost'] - user_budget
        }])
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Predict using the selected best model
        predicted_qualities[grade] = float(model_best.predict(features_scaled)[0])
        
    # 6. Display individual builds
    for grade in ['lower', 'medium', 'best']:
        display_build(grade, final_builds.get(grade), user_budget, predicted_qualities.get(grade, 0.0))
        
    # 7. Print BUILD COMPARISON TABLE
    print("=" * 90)
    print("                                   BUILD COMPARISON TABLE                                 ")
    print("=" * 90)
    print(f"{'Build Grade':<12} | {'Cost':<10} | {'Performance':<12} | {'Budget Diff':<12} | {'Predicted Quality':<18} | {'Build Score':<12}")
    print("-" * 90)
    
    for grade in ['lower', 'medium', 'best']:
        build = final_builds.get(grade)
        if build is None:
            print(f"{grade.upper():<12} | {'N/A':<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<18} | {'N/A':<12}")
        else:
            diff = build['total_cost'] - user_budget
            print(f"{grade.upper():<12} | ${build['total_cost']:<9.2f} | {build['performance_score']:<12.1f} | ${diff:<11.2f} | {predicted_qualities[grade]:<18.1f} | {build['build_score']:<12.1f}")
    print("=" * 90 + "\n")
    
    # 8. Print FINAL AI RECOMMENDATION
    recommended_build = final_builds.get(recommended_grade)
    if recommended_build is not None:
        print("=" * 90)
        print("                                   FINAL AI RECOMMENDATION                                ")
        print("=" * 90)
        print(f"Recommended Build:  {recommended_grade.upper()} GRADE BUILD")
        print(f"Total System Cost:  ${recommended_build['total_cost']:.2f}")
        print(f"Performance Score:  {recommended_build['performance_score']:.1f} / 100")
        print(f"Best Model:         {best_model_name}")
        print(f"Confidence Score:   {confidence:.1f}%")
        print(f"Selection Reason:   {reason}")
        print("=" * 90 + "\n")
        
        # Save run cost/perf graphs
        generate_run_graphs(final_builds, user_budget)
        
    # 9. Print MODEL EVALUATION
    if os.path.exists("models/eval_results.pkl"):
        eval_results = joblib.load("models/eval_results.pkl")
        print("=" * 90)
        print("                               MACHINE LEARNING MODEL EVALUATION                              ")
        print("=" * 90)
        print(f"{'Model Name':<25} | {'Mean Absolute Error (MAE)':<25} | {'Root Mean Squared Error (RMSE)':<30} | {'R2 Score':<10}")
        print("-" * 90)
        for res in eval_results:
            print(f"{res['Model']:<25} | {res['MAE']:<25.3f} | {res['RMSE']:<30.3f} | {res['R2']:<10.3f}")
        print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
