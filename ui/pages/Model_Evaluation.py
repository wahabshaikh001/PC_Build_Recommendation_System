import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Helper function for RMSE compatibility
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.title("Model Evaluation")
    st.write("Compare the accuracy metrics and feature importances of the trained Machine Learning models.")
    
    # Determine project root path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check if files exist
    required_files = [
        os.path.join(project_root, "models", "synthetic_builds.csv"),
        os.path.join(project_root, "models", "scaler.pkl"),
        os.path.join(project_root, "models", "linear_regression.pkl"),
        os.path.join(project_root, "models", "random_forest.pkl"),
        os.path.join(project_root, "models", "xgboost.pkl"),
        os.path.join(project_root, "models", "best_model_info.pkl")
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            st.warning(f"Model training files are missing. Please execute `python train_models.py` to train and evaluate models first.")
            st.stop()
            
    # Load dataset & split (same random state 42 as training pipeline)
    df_builds = pd.read_csv(os.path.join(project_root, "models", "synthetic_builds.csv"))
    features = ['total_cost', 'total_power', 'cpu_score', 'gpu_score', 'ram_score', 
                'storage_score', 'psu_score', 'purpose_encoded', 'compatibility', 'budget_difference']
    
    X = df_builds[features]
    y = df_builds['overall_build_quality']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load scaling & models
    scaler = joblib.load(os.path.join(project_root, "models", "scaler.pkl"))
    model_lr = joblib.load(os.path.join(project_root, "models", "linear_regression.pkl"))
    model_rf = joblib.load(os.path.join(project_root, "models", "random_forest.pkl"))
    model_xgb = joblib.load(os.path.join(project_root, "models", "xgboost.pkl"))
    best_model_info = joblib.load(os.path.join(project_root, "models", "best_model_info.pkl"))
    best_model_name = best_model_info.get('best_model_name', 'XGBoost')
    
    # Scale test set
    X_test_scaled = scaler.transform(X_test)
    
    # Run predictions
    preds_lr = model_lr.predict(X_test_scaled)
    preds_rf = model_rf.predict(X_test_scaled)
    preds_xgb = model_xgb.predict(X_test_scaled)
    
    # Compute metrics
    metrics = {
        'Linear Regression': {
            'MAE': mean_absolute_error(y_test, preds_lr),
            'RMSE': root_mean_squared_error(y_test, preds_lr),
            'R2': r2_score(y_test, preds_lr)
        },
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, preds_rf),
            'RMSE': root_mean_squared_error(y_test, preds_rf),
            'R2': r2_score(y_test, preds_rf)
        },
        'XGBoost': {
            'MAE': mean_absolute_error(y_test, preds_xgb),
            'RMSE': root_mean_squared_error(y_test, preds_xgb),
            'R2': r2_score(y_test, preds_xgb)
        }
    }
    
    # Display highlight indicators
    st.subheader("Best Performing Regressor")
    st.markdown(f"""
    <div class="gradient-card" style="margin-bottom: 25px;">
        <h2 style="margin: 0; color: #8B5CF6 !important;">{best_model_name.upper()} SELECTED</h2>
        <p style="margin: 5px 0 0 0; color: #94A3B8;">Determined programmatically by the build training pipeline based on Highest R² and lowest prediction error.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render table of accuracy metrics
    col_acc1, col_acc2, col_acc3 = st.columns([1, 1, 1])
    
    for idx, (name, val) in enumerate(metrics.items()):
        col = [col_acc1, col_acc2, col_acc3][idx]
        with col:
            st.markdown(f"""
            <div class="premium-card">
                <h4>{name}</h4>
                <hr style="border-color: rgba(255, 255, 255, 0.05); margin: 8px 0;">
                <p style="margin: 4px 0;"><b>MAE:</b> <span class="accent-cyan">{val['MAE']:.3f}</span></p>
                <p style="margin: 4px 0;"><b>RMSE:</b> <span class="accent-purple">{val['RMSE']:.3f}</span></p>
                <p style="margin: 4px 0;"><b>R² Score:</b> <span class="accent-blue">{val['R2']:.3f}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    
    # ================= PLOT 1: Model Comparison Chart =================
    st.subheader("Model Comparison Chart")
    
    df_eval = pd.DataFrame([
        {'Model': k, 'MAE': v['MAE'], 'RMSE': v['RMSE'], 'R2': v['R2']} for k, v in metrics.items()
    ])
    
    tab_error, tab_r2 = st.tabs(["Error Metrics (MAE / RMSE)", "R² Score"])
    
    with tab_error:
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=df_eval['Model'], y=df_eval['MAE'], name='MAE',
            marker_color='#3B82F6', marker_line_color='black', marker_line_width=1
        ))
        fig_err.add_trace(go.Bar(
            x=df_eval['Model'], y=df_eval['RMSE'], name='RMSE',
            marker_color='#EF4444', marker_line_color='black', marker_line_width=1
        ))
        fig_err.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.5)',
            font={'color': "#F8FAFC", 'family': "Inter"},
            yaxis_title="Error Value (Lower is Better)",
            margin=dict(l=40, r=40, t=20, b=40),
            height=350,
            grid={'rows': 1, 'columns': 1}
        )
        st.plotly_chart(fig_err, width='stretch')
        
    with tab_r2:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=df_eval['Model'], y=df_eval['R2'], name='R2 Score',
            marker_color='#10B981', marker_line_color='black', marker_line_width=1,
            width=0.4
        ))
        fig_r2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.5)',
            font={'color': "#F8FAFC", 'family': "Inter"},
            yaxis_title="R2 Score (Higher is Better)",
            yaxis_range=[0, 1.1],
            margin=dict(l=40, r=40, t=20, b=40),
            height=350
        )
        st.plotly_chart(fig_r2, width='stretch')

    # ================= PLOT 2 & 3: Feature Importances =================
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    st.subheader("Model Feature Importances")
    
    col_feat1, col_feat2 = st.columns([1, 1], gap="medium")
    
    with col_feat1:
        st.markdown("##### Random Forest Feature Importance")
        importances_rf = model_rf.feature_importances_
        df_rf_imp = pd.DataFrame({'Feature': features, 'Importance': importances_rf}).sort_values(by='Importance', ascending=True)
        
        fig_rf = px.bar(
            df_rf_imp, x='Importance', y='Feature', orientation='h',
            color_discrete_sequence=['#8B5CF6']
        )
        fig_rf.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.5)',
            font={'color': "#F8FAFC", 'family': "Inter"},
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis_title="Importance value"
        )
        st.plotly_chart(fig_rf, width='stretch')
        
    with col_feat2:
        st.markdown("##### XGBoost Feature Importance")
        importances_xgb = model_xgb.feature_importances_
        df_xgb_imp = pd.DataFrame({'Feature': features, 'Importance': importances_xgb}).sort_values(by='Importance', ascending=True)
        
        fig_xgb = px.bar(
            df_xgb_imp, x='Importance', y='Feature', orientation='h',
            color_discrete_sequence=['#06B6D4']
        )
        fig_xgb.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.5)',
            font={'color': "#F8FAFC", 'family': "Inter"},
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis_title="Importance value"
        )
        st.plotly_chart(fig_xgb, width='stretch')

    # ================= PLOT 4 & 5: Prediction vs Actual & Residuals =================
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    st.subheader("Prediction Error Visualizations")
    
    col_err1, col_err2 = st.columns([1, 1], gap="medium")
    
    # Sample 400 points to keep plots fast and clean
    sample_indices = np.random.choice(len(y_test), min(400, len(y_test)), replace=False)
    y_test_sample = y_test.iloc[sample_indices]
    preds_sample_xgb = preds_xgb[sample_indices]
    
    with col_err1:
        st.markdown("##### XGBoost: Prediction vs Actual Quality")
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_test_sample, y=preds_sample_xgb, mode='markers',
            marker=dict(color='#EF4444', size=7, opacity=0.7, line=dict(width=0.5, color='black')),
            name='Predictions'
        ))
        
        # Perfect fit diagonal
        min_val = min(y_test_sample.min(), preds_sample_xgb.min())
        max_val = max(y_test_sample.max(), preds_sample_xgb.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color='white', width=1.5, dash='dash'),
            name='Perfect Fit'
        ))
        
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.5)',
            font={'color': "#F8FAFC", 'family': "Inter"},
            xaxis_title="Actual Quality Score",
            yaxis_title="Predicted Quality Score",
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_scatter, width='stretch')
        
    with col_err2:
        st.markdown("##### XGBoost: Residual Error Plot")
        residuals = y_test_sample - preds_sample_xgb
        
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=preds_sample_xgb, y=residuals, mode='markers',
            marker=dict(color='#F5B041', size=7, opacity=0.7, line=dict(width=0.5, color='black')),
            name='Residuals'
        ))
        # Horizontal reference line at residual = 0
        fig_resid.add_trace(go.Scatter(
            x=[preds_sample_xgb.min(), preds_sample_xgb.max()], y=[0, 0],
            mode='lines', line=dict(color='red', width=1.5, dash='dash'),
            name='Zero Line'
        ))
        
        fig_resid.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.5)',
            font={'color': "#F8FAFC", 'family': "Inter"},
            xaxis_title="Predicted Quality Score",
            yaxis_title="Residual (Actual - Predicted)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_resid, width='stretch')

if __name__ == "__main__":
    main()
