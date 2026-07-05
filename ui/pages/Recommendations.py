import streamlit as st
import os
import sys
import plotly.graph_objects as go

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_build_card(grade, build, predicted_quality):
    """
    Renders a clean card containing cost, performance, build score, and specs.
    """
    if build is None:
        st.markdown(f"""
        <div class="premium-card">
            <h3 style="color: #EF4444 !important;">{grade.upper()} GRADE BUILD</h3>
            <p class="text-muted">Could not generate a compatible build within this budget range.</p>
        </div>
        """, unsafe_allow_html=True)
        return
        
    st.markdown(f"""
    <div class="premium-card">
        <h3 class="accent-blue">{grade.upper()} GRADE</h3>
        <hr style="border-color: rgba(255, 255, 255, 0.05); margin: 10px 0;">
        <p style="font-size: 1.5rem; font-weight: 700; margin: 5px 0;">${build['total_cost']:.2f}</p>
        <p style="margin: 2px 0;"><b>Performance:</b> {build['performance_score']:.1f} / 100</p>
        <p style="margin: 2px 0;"><b>Build Score:</b> {build['build_score']:.1f} / 100</p>
        <p style="margin: 2px 0;"><b>Predicted Quality:</b> {predicted_quality:.1f} / 100</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render specifications in streamlit table/dataframe format for clean structure
    st.markdown("##### Specifications")
    st.write(f"**CPU:** {build['cpu']['name']} (${build['cpu']['price']:.2f})")
    st.write(f"**GPU:** {build['gpu']['name']} (${build['gpu']['price']:.2f})")
    st.write(f"**RAM:** {build['ram']['name']} (${build['ram']['price']:.2f})")
    st.write(f"**Mobo:** {build['motherboard']['name']} (${build['motherboard']['price']:.2f})")
    st.write(f"**Storage:** {build['storage']['name']} (${build['storage']['price']:.2f})")
    st.write(f"**PSU:** {build['psu']['name']} (${build['psu']['price']:.2f})")
    psu_modular = build['psu'].get('modular', 'Unknown')
    modularity = "Fully Modular" if psu_modular == "Yes" else "Non-Modular" if psu_modular == "No" else psu_modular
    st.write(f"**PSU Modularity:** {modularity}")

def generate_why_explanations(build, purpose):
    """
    Generates component-level explanations dynamically from characteristics.
    """
    cpu = build['cpu']
    gpu = build['gpu']
    ram = build['ram']
    mobo = build['motherboard']
    storage = build['storage']
    psu = build['psu']
    
    explanations = {
        'cpu': f"Selected because it features the {cpu['socket']} socket matching the motherboard and offers {int(cpu['cores'])} cores / {int(cpu['threads'])} threads with a boost clock of {cpu['boost_clock']} GHz to handle heavy processing workloads.",
        'gpu': f"Chosen because it delivers {gpu['benchmark_score']:.0f} points of benchmark performance with {int(gpu['vram'])} GB VRAM, providing excellent rendering power matching the {purpose} requirements.",
        'ram': f"Selected matching the motherboard's {mobo['ram_type']} type at a high speed of {int(ram['speed'])} MHz to eliminate bandwidth bottlenecks.",
        'motherboard': f"Chosen because its {mobo['socket']} socket and {mobo['chipset']} chipset fully support the CPU while enabling memory expansion up to {int(mobo['max_ram'])} GB.",
        'storage': f"Selected because its {storage['capacity']} capacity and {storage['speed']} transfer speed satisfy both data storage and low system-latency requirements.",
        'psu': f"Chosen because it delivers {int(psu['wattage'])}W of power with an efficiency rating of '{psu['efficiency_rating']}', safely satisfying system demands with a 30% overhead margin."
    }
    return explanations

def main():
    st.title("PC Build Recommendations")
    st.write("Review the generated configurations and check the recommendation details.")
    
    # 1. Verification of session state
    if 'recommendations' not in st.session_state:
        st.warning("No recommendation data found. Please set your preferences on the Home Page first.")
        if st.button("Back to Home"):
            if 'home_page' in st.session_state:
                st.switch_page(st.session_state['home_page'])
            else:
                st.switch_page("app.py")
        st.stop()
        
    recommendations = st.session_state['recommendations']
    recommended_grade = st.session_state['recommended_grade']
    confidence = st.session_state['confidence']
    reason = st.session_state['reason']
    predicted_qualities = st.session_state['predicted_qualities']
    user_budget = st.session_state['user_budget']
    purpose = st.session_state['purpose']
    best_model_name = st.session_state['best_model_name']
    
    if recommended_grade is None or not recommendations:
        st.error("Could not find any compatible configurations for your budget.")
        if st.button("Back to Home"):
            if 'home_page' in st.session_state:
                st.switch_page(st.session_state['home_page'])
            else:
                st.switch_page("app.py")
        st.stop()

    # Three Tier columns
    st.subheader("Build Comparison")
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    with col1:
        render_build_card('lower', recommendations.get('lower'), predicted_qualities.get('lower', 0.0))
        
    with col2:
        render_build_card('medium', recommendations.get('medium'), predicted_qualities.get('medium', 0.0))
        
    with col3:
        render_build_card('best', recommendations.get('best'), predicted_qualities.get('best', 0.0))

    st.markdown('<div style="margin-top: 40px;"></div>', unsafe_allow_html=True)
    st.subheader("Recommended Build")
    
    rec_build = recommendations[recommended_grade]
    
    # Highlighted card and Gauge Side-by-Side
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        st.markdown(f"""
        <div class="gradient-card">
            <h2 style="margin: 0; color: #8B5CF6 !important;">RECOMMENDED BUILD: {recommended_grade.upper()} GRADE BUILD</h2>
            <hr style="border-color: rgba(255, 255, 255, 0.1); margin: 15px 0;">
            <p style="font-size: 1.2rem; margin: 5px 0;"><b>Total Build Cost:</b> <span class="accent-cyan">${rec_build['total_cost']:.2f}</span></p>
            <p style="font-size: 1.2rem; margin: 5px 0;"><b>Confidence Score:</b> <span class="accent-blue">{confidence:.1f}%</span></p>
            <p style="font-size: 1.2rem; margin: 5px 0;"><b>ML Predictor:</b> <span class="text-muted">{best_model_name}</span></p>
            <p style="font-size: 1.2rem; margin: 5px 0;"><b>Quality Prediction:</b> <span class="accent-cyan">{predicted_qualities[recommended_grade]:.1f} / 100</span></p>
            <p style="font-size: 1.1rem; line-height: 1.5; background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 8px; margin-top: 15px;">
                <b>Reason:</b> {reason}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        with st.container(key="glass-panel-recs-right"):
            st.subheader("Budget Utilization")
            
            # Plotly budget utilisation gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = rec_build['total_cost'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                delta = {'reference': user_budget, 'position': "top", 'relative': False, 'valueformat': "+$.2f"},
                title = {'text': "Cost vs User Budget", 'font': {'size': 16, 'color': "#F8FAFC"}},
                gauge = {
                    'axis': {'range': [None, user_budget * 1.3], 'tickwidth': 1, 'tickcolor': "#F8FAFC"},
                    'bar': {'color': "#8B5CF6"},
                    'bgcolor': "#1F2937",
                    'borderwidth': 2,
                    'bordercolor': "#4B5563",
                    'steps': [
                        {'range': [0, user_budget * 0.85], 'color': 'rgba(6, 182, 212, 0.2)'},
                        {'range': [user_budget * 0.85, user_budget], 'color': 'rgba(59, 130, 246, 0.2)'},
                        {'range': [user_budget, user_budget * 1.3], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': user_budget
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#F8FAFC", 'family': "Inter"},
                height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, width='stretch')
 
    st.markdown('<div style="margin-top: 40px;"></div>', unsafe_allow_html=True)
    st.subheader("Performance Analysis")
    
    why_exps = generate_why_explanations(rec_build, purpose)
    
    col_why1, col_why2 = st.columns([1, 1], gap="medium")
    
    with col_why1:
        st.markdown(f"""
        <div class="glass-panel">
            <h4 class="accent-blue">CPU Selection</h4>
            <p style="font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;">{why_exps['cpu']}</p>
        </div>
        <div class="glass-panel">
            <h4 class="accent-purple">GPU Selection</h4>
            <p style="font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;">{why_exps['gpu']}</p>
        </div>
        <div class="glass-panel">
            <h4 class="accent-cyan">RAM Selection</h4>
            <p style="font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;">{why_exps['ram']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_why2:
        st.markdown(f"""
        <div class="glass-panel">
            <h4 class="accent-blue">Motherboard Selection</h4>
            <p style="font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;">{why_exps['motherboard']}</p>
        </div>
        <div class="glass-panel">
            <h4 class="accent-purple">Storage Selection</h4>
            <p style="font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;">{why_exps['storage']}</p>
        </div>
        <div class="glass-panel">
            <h4 class="accent-cyan">PSU Selection</h4>
            <p style="font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;">{why_exps['psu']}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
