import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Add project root to path for direct script runs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Helper function for RMSE compatibility across sklearn versions
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def main():
    print("Starting Machine Learning model evaluation...")
    
    # Ensure graph directory exists
    os.makedirs("results/graphs", exist_ok=True)
    
    # 1. Load data
    if not os.path.exists("models/synthetic_builds.csv"):
        raise FileNotFoundError("Could not find synthetic_builds.csv. Please run train_models.py first.")
        
    df_builds = pd.read_csv("models/synthetic_builds.csv")
    
    # 2. Extract features and target
    features = ['total_cost', 'total_power', 'cpu_score', 'gpu_score', 'ram_score', 
                'storage_score', 'psu_score', 'purpose_encoded', 'compatibility', 'budget_difference']
    X = df_builds[features]
    y = df_builds['overall_build_quality']
    
    # Train-test split (same random state as train_models.py)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Load scaler and models
    scaler = joblib.load("models/scaler.pkl")
    model_lr = joblib.load("models/linear_regression.pkl")
    model_rf = joblib.load("models/random_forest.pkl")
    model_xgb = joblib.load("models/xgboost.pkl")
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Predict and evaluate
    models = {
        'Linear Regression': model_lr,
        'Random Forest': model_rf,
        'XGBoost': model_xgb
    }
    
    eval_results = []
    predictions = {}
    
    print("\nModel Performance Summary:")
    print("=" * 60)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        preds = model.predict(X_test_scaled)
        predictions[name] = preds
        
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        eval_results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        
        print(f"{name:<20} | {mae:<10.3f} | {rmse:<10.3f} | {r2:<10.3f}")
        
    df_eval = pd.DataFrame(eval_results)
    
    # Save evaluation results for display in main.py
    joblib.dump(eval_results, "models/eval_results.pkl")
    
    # ================= VISUALIZATION 1: Model Comparison Chart =================
    print("\nGenerating Chart 1: Model Comparison Chart...")
    plt.figure(figsize=(12, 6))
    
    # Subplot for MAE & RMSE, Subplot for R2
    plt.subplot(1, 2, 1)
    x = np.arange(len(df_eval))
    width = 0.35
    
    plt.bar(x - width/2, df_eval['MAE'], width, label='MAE', color='#3498db', edgecolor='black', alpha=0.9)
    plt.bar(x + width/2, df_eval['RMSE'], width, label='RMSE', color='#e74c3c', edgecolor='black', alpha=0.9)
    plt.ylabel('Error Value')
    plt.title('Error Metrics (Lower is Better)')
    plt.xticks(x, df_eval['Model'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.bar(df_eval['Model'], df_eval['R2'], width * 1.5, color='#2ecc71', edgecolor='black', alpha=0.9)
    plt.ylabel('R2 Score')
    plt.title('R2 Score (Higher is Better)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.suptitle('Model Evaluation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/graphs/model_comparison.png', dpi=150)
    plt.close()
    
    # ================= VISUALIZATION 2: Random Forest Feature Importance =================
    print("Generating Chart 2: Random Forest Feature Importance...")
    importances_rf = model_rf.feature_importances_
    df_rf_imp = pd.DataFrame({'Feature': features, 'Importance': importances_rf}).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 5))
    plt.barh(df_rf_imp['Feature'], df_rf_imp['Importance'], color='#6C5B7B', edgecolor='black')
    plt.xlabel('Importance Value')
    plt.title('Random Forest Feature Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/graphs/random_forest_importance.png', dpi=150)
    plt.close()
    
    # ================= VISUALIZATION 3: XGBoost Feature Importance =================
    print("Generating Chart 3: XGBoost Feature Importance...")
    importances_xgb = model_xgb.feature_importances_
    df_xgb_imp = pd.DataFrame({'Feature': features, 'Importance': importances_xgb}).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 5))
    plt.barh(df_xgb_imp['Feature'], df_xgb_imp['Importance'], color='#355C7D', edgecolor='black')
    plt.xlabel('Importance Value')
    plt.title('XGBoost Feature Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/graphs/xgboost_importance.png', dpi=150)
    plt.close()
    
    # ================= VISUALIZATION 4: Prediction vs Actual =================
    print("Generating Chart 4: Prediction vs Actual...")
    # Sample 500 points to make scatter plot readable
    sample_indices = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
    y_test_sample = y_test.iloc[sample_indices]
    preds_sample = predictions['XGBoost'][sample_indices]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_sample, preds_sample, color='#E74C3C', alpha=0.6, edgecolors='black', label='Predicted vs Actual')
    # Perfect fit line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k--', alpha=0.8, zorder=3, label='Perfect Fit (Y = X)')
    plt.xlabel('Actual Quality Score')
    plt.ylabel('Predicted Quality Score')
    plt.title('XGBoost: Prediction vs Actual Quality')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/graphs/prediction_vs_actual.png', dpi=150)
    plt.close()
    
    # ================= VISUALIZATION 5: Residual Error Plot =================
    print("Generating Chart 5: Residual Error Plot...")
    residuals = y_test_sample - preds_sample
    
    plt.figure(figsize=(10, 5))
    plt.scatter(preds_sample, residuals, color='#F8B195', alpha=0.7, edgecolors='black')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    plt.xlabel('Predicted Quality Score')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('XGBoost: Residual Error Plot')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/graphs/residual_error.png', dpi=150)
    plt.close()
    
    print("\nVisualizations successfully generated and saved to results/graphs/ folder!")

if __name__ == "__main__":
    main()
