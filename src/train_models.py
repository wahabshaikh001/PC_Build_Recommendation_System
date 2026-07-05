import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# Helper function for RMSE compatibility
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Add project root to path for direct script runs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_datasets
from src.preprocessing import preprocess_datasets
from src.feature_engineering import add_scores_to_datasets
from src.compatibility_engine import check_compatibility
from src.budget_engine import get_component_budget_allocation
from src.performance_engine import calculate_build_performance

def generate_synthetic_builds(datasets, max_scores, num_builds=10000):
    """
    Generates a diverse build-level dataset of compatible and incompatible builds
    with a synthetic 'overall_build_quality' target.
    """
    np.random.seed(42)
    
    cpus = datasets['cpu'].to_dict('records')
    gpus = datasets['gpu'].to_dict('records')
    rams = datasets['ram'].to_dict('records')
    mobos = datasets['motherboard'].to_dict('records')
    storages = datasets['storage'].to_dict('records')
    psus = datasets['psu'].to_dict('records')
    
    purposes = ['Gaming', 'Editing', 'Office']
    
    builds_data = []
    
    for _ in range(num_builds):
        # 1. Random user budget between 400 and 4000
        user_budget = float(np.random.uniform(400, 4000))
        
        # 2. Random purpose
        purpose = np.random.choice(purposes)
        
        # 3. Randomly sample components
        cpu = cpus[np.random.randint(len(cpus))]
        gpu = gpus[np.random.randint(len(gpus))]
        ram = rams[np.random.randint(len(rams))]
        mobo = mobos[np.random.randint(len(mobos))]
        storage = storages[np.random.randint(len(storages))]
        psu = psus[np.random.randint(len(psus))]
        
        # 4. Check compatibility
        is_compatible, _ = check_compatibility(cpu, gpu, ram, mobo, psu)
        compatibility_val = 1.0 if is_compatible else 0.0
        
        # 5. Calculate costs, power, scores
        total_cost = float(cpu['price'] + gpu['price'] + ram['price'] + mobo['price'] + storage['price'] + psu['price'])
        total_power = float(cpu['tdp'] + gpu['power_draw'])
        budget_difference = total_cost - user_budget
        
        # 6. Calculate Build Score components
        perf_score = calculate_build_performance(
            cpu, gpu, ram, mobo, storage, psu, purpose, max_scores, total_cost, user_budget
        )
        
        budget_efficiency = max(0.0, 100.0 * (1.0 - abs(total_cost - user_budget) / user_budget))
        
        # Purpose Match using similarity (approximated for training generation)
        allocations = get_component_budget_allocation(purpose)
        
        t_cpu = allocations['cpu'] * user_budget
        t_gpu = allocations['gpu'] * user_budget
        t_ram = allocations['ram'] * user_budget
        t_mobo = allocations['motherboard'] * user_budget
        t_storage = allocations['storage'] * user_budget
        t_psu = allocations['psu'] * user_budget
        
        cpu_sim = max(0.0, 1.0 - abs(cpu['price'] - t_cpu) / t_cpu)
        gpu_sim = max(0.0, 1.0 - abs(gpu['price'] - t_gpu) / t_gpu)
        ram_sim = max(0.0, 1.0 - abs(ram['price'] - t_ram) / t_ram)
        mobo_sim = max(0.0, 1.0 - abs(mobo['price'] - t_mobo) / t_mobo)
        storage_sim = max(0.0, 1.0 - abs(storage['price'] - t_storage) / t_storage)
        psu_sim = max(0.0, 1.0 - abs(psu['price'] - t_psu) / t_psu)
        
        avg_similarity = (cpu_sim + gpu_sim + ram_sim + mobo_sim + storage_sim + psu_sim) / 6.0
        purpose_match_score = avg_similarity * 100.0
        compatibility_score = compatibility_val * 100.0
        
        # Raw build score
        raw_build_score = (0.40 * perf_score + 
                           0.30 * budget_efficiency + 
                           0.20 * purpose_match_score + 
                           0.10 * compatibility_score)
        
        # Apply budget penalty
        penalty = 0.0
        if total_cost > user_budget:
            ratio_over = (total_cost - user_budget) / user_budget
            penalty = 50.0 * ratio_over + 100.0 * (ratio_over ** 2)
            
        overall_build_quality = max(0.0, raw_build_score - penalty)
        
        # Add random noise to make ML learning realistic
        noise = np.random.normal(0.0, 1.5)
        overall_build_quality = float(np.clip(overall_build_quality + noise, 0.0, 100.0))
        
        purpose_map = {'Gaming': 0, 'Editing': 1, 'Office': 2}
        
        builds_data.append({
            'total_cost': total_cost,
            'total_power': total_power,
            'cpu_score': cpu['cpu_score'],
            'gpu_score': gpu['gpu_score'],
            'ram_score': ram['ram_score'],
            'storage_score': storage['storage_score'],
            'psu_score': psu['psu_score'],
            'purpose': purpose,
            'purpose_encoded': purpose_map[purpose],
            'compatibility': compatibility_val,
            'budget_difference': budget_difference,
            'overall_build_quality': overall_build_quality
        })
        
    return pd.DataFrame(builds_data)

def main():
    print("Starting Machine Learning model training pipeline...")
    
    # 1. Load data
    print("Loading datasets...")
    datasets = load_datasets()
    
    # Track raw records count loaded
    cpu_count = len(datasets['cpu'])
    gpu_count = len(datasets['gpu'])
    ram_count = len(datasets['ram'])
    mobo_count = len(datasets['motherboard'])
    storage_count = len(datasets['storage'])
    psu_count = len(datasets['psu'])
    
    # 2. Preprocess data (fits label encoders and saves to models/)
    print("Preprocessing components...")
    datasets = preprocess_datasets(datasets)
    
    # 3. Add engineered features
    print("Engineering component features...")
    datasets = add_scores_to_datasets(datasets)
    
    # Calculate dataset-wide max scores
    max_scores = {
        'cpu': datasets['cpu']['cpu_score'].max(),
        'gpu': datasets['gpu']['gpu_score'].max(),
        'ram': datasets['ram']['ram_score'].max(),
        'storage': datasets['storage']['storage_score'].max(),
        'psu': datasets['psu']['psu_score'].max(),
        'motherboard': datasets['motherboard']['max_ram'].max()
    }
    
    # Save max scores to models for use in recommendation engine
    os.makedirs("models", exist_ok=True)
    joblib.dump(max_scores, "models/max_scores.pkl")
    
    # 4. Generate build quality dataset
    print("Generating synthetic build dataset (10,000 builds)...")
    df_builds = generate_synthetic_builds(datasets, max_scores, num_builds=10000)
    df_builds.to_csv("models/synthetic_builds.csv", index=False)
    
    # 5. Set up Purpose encoder and update label encoders
    print("Updating label encoders with Purpose encoder...")
    encoders = joblib.load("models/label_encoders.pkl")
    le_purpose = LabelEncoder()
    le_purpose.fit(['Gaming', 'Editing', 'Office'])
    encoders['purpose'] = le_purpose
    joblib.dump(encoders, "models/label_encoders.pkl")
    
    # 6. Feature scaling
    features = ['total_cost', 'total_power', 'cpu_score', 'gpu_score', 'ram_score', 
                'storage_score', 'psu_score', 'purpose_encoded', 'compatibility', 'budget_difference']
    X = df_builds[features]
    y = df_builds['overall_build_quality']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale build features
    print("Fitting and saving build features StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # 7. Train Models
    print("Training Linear Regression...")
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    joblib.dump(model_lr, "models/linear_regression.pkl")
    
    print("Training Random Forest...")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    joblib.dump(model_rf, "models/random_forest.pkl")
    
    print("Training XGBoost Regressor...")
    model_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
    model_xgb.fit(X_train_scaled, y_train)
    joblib.dump(model_xgb, "models/xgboost.pkl")
    
    # 8. Compare all models to find the best performing one
    trained_models = {
        'Linear Regression': model_lr,
        'Random Forest': model_rf,
        'XGBoost': model_xgb
    }
    
    metrics = {}
    for name, model in trained_models.items():
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
    # Programmatic selection: sort by highest R2, then lowest MAE, then lowest RMSE
    best_model_name = max(metrics.keys(), key=lambda k: (metrics[k]['r2'], -metrics[k]['mae'], -metrics[k]['rmse']))
    
    # Save best model information and clone best model file
    joblib.dump({'best_model_name': best_model_name}, "models/best_model_info.pkl")
    joblib.dump(trained_models[best_model_name], "models/best_model.pkl")
    
    # 9. Print Training Summary
    print("\n" + "=" * 60)
    print("                       TRAINING SUMMARY                       ")
    print("=" * 60)
    print(f"CPU Records Loaded:         {cpu_count}")
    print(f"GPU Records Loaded:         {gpu_count}")
    print(f"RAM Records Loaded:         {ram_count}")
    print(f"Motherboard Records Loaded: {mobo_count}")
    print(f"Storage Records Loaded:     {storage_count}")
    print(f"PSU Records Loaded:         {psu_count}")
    print("-" * 60)
    print(f"Best Model Selected:        {best_model_name}")
    print("-" * 60)
    print("Model Accuracy Metrics:")
    print(f"  - Linear Regression: MAE = {metrics['Linear Regression']['mae']:.3f}, RMSE = {metrics['Linear Regression']['rmse']:.3f}, R2 = {metrics['Linear Regression']['r2']:.3f}")
    print(f"  - Random Forest:     MAE = {metrics['Random Forest']['mae']:.3f}, RMSE = {metrics['Random Forest']['rmse']:.3f}, R2 = {metrics['Random Forest']['r2']:.3f}")
    print(f"  - XGBoost:           MAE = {metrics['XGBoost']['mae']:.3f}, RMSE = {metrics['XGBoost']['rmse']:.3f}, R2 = {metrics['XGBoost']['r2']:.3f}")
    print("-" * 60)
    print("PKL Files Saved Successfully: Yes")
    print("=" * 60 + "\n")
    
    # Run evaluation & generate graphs
    print("Running evaluation and generating graphs...")
    from src.evaluate_models import main as evaluate_main
    evaluate_main()

if __name__ == "__main__":
    main()

