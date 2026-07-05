import streamlit as st
import os
import sys
import joblib
import pandas as pd

# Add the project root to sys.path so we can import src.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration at the entrypoint level (no emojis in title or icon)
st.set_page_config(
    page_title="CoreAI - PC Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS stylesheet at the top level so it applies to all navigated pages
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Imports from src
from src.data_loader import load_datasets
from src.preprocessing import preprocess_datasets
from src.feature_engineering import add_scores_to_datasets
from src.build_generator import generate_builds_for_target
from src.recommendation_engine import get_final_recommendation

@st.cache_data
def get_cached_datasets():
    datasets = load_datasets()
    datasets = preprocess_datasets(datasets)
    datasets = add_scores_to_datasets(datasets)
    return datasets

# Define file-based pages (must be before render_home so it can reference recs_page)
recs_page = st.Page("pages/Recommendations.py", title="Recommendations", url_path="Recommendations")
eval_page = st.Page("pages/Model_Evaluation.py", title="Model Evaluation", url_path="Model_Evaluation")
insights_page = st.Page("pages/Dataset_Insights.py", title="Dataset Insights", url_path="Dataset_Insights")

def render_home():
    # Hero Header Section
    st.markdown('<div class="hero-title">CoreAI Build Engine</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94A3B8; font-size: 1.2rem; margin-bottom: 2rem;">Build your dream PC powered by Machine Learning and Cosine Similarity.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(key="glass-panel-home-left"):
            st.subheader("Build Preferences")
            st.write("Set your budget and usage purpose to generate recommendations.")
            
            user_budget = st.number_input("Budget (USD)", min_value=0, max_value=10000, value=None, step=100, placeholder="Enter budget (e.g. 1500)")
            
            purpose = st.selectbox(
                "PC Build Purpose",
                options=["Gaming", "Editing", "Office"],
                index=0
            )
            
            st.write("")
            generate_btn = st.button("Generate Recommendation")

    with col2:
        with st.container(key="glass-panel-home-right"):
            st.subheader("Engine Architecture")
            st.markdown("""
            Our dynamic recommendation pipeline processes components using:
            
            - **Preprocessing & Score Engineering**: Dynamically scales and parses component features.
            - **Cosine Similarity Engine**: Compares components against ideal target profiles (CPU, GPU, RAM, etc.).
            - **Compatibility Engine**: Enforces strict socket alignment, memory types, and power overhead safety checks.
            - **XGBoost Build Quality Prediction**: Automatically evaluates final system builds using our best-performing trained ML model.
            - **Budget Penalty System**: Ensures balanced choices by penalizing configurations that exceed your budget.
            """)

    # Trigger Generation
    if generate_btn:
        if user_budget is None:
            st.error("Please enter a budget.")
        elif user_budget <= 0:
            st.error("Please enter a budget greater than 0.")
        elif user_budget > 2000:
            st.error("Please reduce your budget to $2000 or less.")
        else:
            with st.spinner("Running similarity search and compatibility solver..."):
                try:
                    # 1. Check if model files exist
                    required_files = [
                        "models/scaler.pkl", "models/best_model.pkl", 
                        "models/best_model_info.pkl", "models/max_scores.pkl"
                    ]
                    for f in required_files:
                        if not os.path.exists(f):
                            st.error(f"Missing required model weight '{f}'. Please run 'python train_models.py' first.")
                            st.stop()
                            
                    # 2. Load component datasets
                    datasets = get_cached_datasets()
                    
                    # 3. Load model parameters
                    max_scores = joblib.load("models/max_scores.pkl")
                    scaler = joblib.load("models/scaler.pkl")
                    model_best = joblib.load("models/best_model.pkl")
                    best_model_info = joblib.load("models/best_model_info.pkl")
                    best_model_name = best_model_info.get('best_model_name', 'XGBoost')
                    
                    # 4. Target Budgets
                    target_budgets = {
                        'lower': 0.85 * user_budget,
                        'medium': 1.00 * user_budget,
                        'best': 1.10 * user_budget
                    }
                    
                    # 5. Generate Builds
                    raw_builds = {}
                    for grade, target in target_budgets.items():
                        raw_builds[grade] = generate_builds_for_target(datasets, purpose, target, max_scores, user_budget)
                        
                    # 6. Final Recommendation and Penalty engine
                    recommended_grade, final_builds, confidence, reason = get_final_recommendation(
                        raw_builds, user_budget, purpose, max_scores
                    )
                    
                    # 7. Predict overall build quality
                    predicted_qualities = {}
                    purpose_encoded_map = {'Gaming': 0, 'Editing': 1, 'Office': 2}
                    
                    for grade, build in final_builds.items():
                        if build is None:
                            continue
                        features_df = pd.DataFrame([{
                            'total_cost': build['total_cost'],
                            'total_power': build['total_power'],
                            'cpu_score': build['cpu_score'],
                            'gpu_score': build['gpu_score'],
                            'ram_score': build['ram_score'],
                            'storage_score': build['storage_score'],
                            'psu_score': build['psu_score'],
                            'purpose_encoded': purpose_encoded_map[purpose],
                            'compatibility': 1.0,
                            'budget_difference': build['total_cost'] - user_budget
                        }])
                        features_scaled = scaler.transform(features_df)
                        predicted_qualities[grade] = float(model_best.predict(features_scaled)[0])
                        
                    # 8. Store in session state for other pages
                    st.session_state['recommendations'] = final_builds
                    st.session_state['recommended_grade'] = recommended_grade
                    st.session_state['confidence'] = confidence
                    st.session_state['reason'] = reason
                    st.session_state['predicted_qualities'] = predicted_qualities
                    st.session_state['user_budget'] = user_budget
                    st.session_state['purpose'] = purpose
                    st.session_state['best_model_name'] = best_model_name
                    
                    st.success("Recommendations successfully generated! Redirecting...")
                    st.switch_page(recs_page)
                except Exception as e:
                    st.error(f"Failed to generate builds: {e}")

# Navigation Definition
home_page = st.Page(render_home, title="Home", url_path="home", default=True)

pg = st.navigation([home_page, recs_page, eval_page, insights_page])
pg.run()
