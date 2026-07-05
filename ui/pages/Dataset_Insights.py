import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to sys.path so we can import src.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load CSS stylesheet
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Imports from src
from src.data_loader import load_datasets
from src.preprocessing import preprocess_datasets
from src.feature_engineering import add_scores_to_datasets

@st.cache_data
def get_cached_datasets():
    datasets = load_datasets()
    datasets = preprocess_datasets(datasets)
    datasets = add_scores_to_datasets(datasets)
    return datasets

def main():
    st.title("Dataset Insights")
    st.write("Explore and visualize the loaded component datasets dynamically from the DATASET folder.")

    # Load datasets
    with st.spinner("Loading component datasets..."):
        try:
            datasets = get_cached_datasets()
        except Exception as e:
            st.error(f"Failed to load datasets: {e}")
            st.stop()

    # 1. Dataset Counts / KPI metrics
    st.subheader("Database Records Overview")
    cols = st.columns(6)
    
    component_names = ['cpu', 'gpu', 'ram', 'motherboard', 'storage', 'psu']
    display_names = ['CPUs', 'GPUs', 'RAM Kits', 'Motherboards', 'Storage Drives', 'PSUs']
    colors = ['accent-blue', 'accent-purple', 'accent-cyan', 'accent-blue', 'accent-purple', 'accent-cyan']

    for i, comp in enumerate(component_names):
        count = len(datasets[comp])
        cols[i].markdown(f"""
        <div class="premium-card" style="text-align: center; padding: 20px; margin-bottom: 10px;">
            <div style="font-size: 0.95rem; color: #94A3B8; font-weight: 500;">{display_names[i]}</div>
            <div class="{colors[i]}" style="font-size: 2.0rem; font-weight: 800; margin-top: 5px;">{count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 35px;"></div>', unsafe_allow_html=True)

    # 2. Main tabs (emojis removed)
    tab_overview, tab_prices, tab_cpus_mobos, tab_gpus, tab_ram_storage, tab_psus = st.tabs([
        "System Overview", 
        "Price Analysis", 
        "CPU & Motherboard", 
        "GPU Performance", 
        "Memory & Storage", 
        "PSU Efficiency"
    ])

    # Plot layout colors
    plotly_layout_args = {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(15,23,42,0.5)',
        'font': {'color': "#F8FAFC", 'family': "Inter"},
        'margin': dict(l=40, r=40, t=30, b=40)
    }

    # ================= TAB 1: System Overview =================
    with tab_overview:
        st.subheader("Inventory and Component Breakdown")
        
        # Add a nice description and overview charts
        col_ov1, col_ov2 = st.columns([1, 1], gap="large")
        
        with col_ov1:
            with st.container(key="glass-panel-insights-left"):
                st.markdown("##### Socket Type Distribution")
                st.write("Socket compatibility constraints form the base of the build solver. Here is the socket count overlap:")
                
                # Combine CPU and Mobo socket counts
                cpu_sockets = datasets['cpu']['socket'].value_counts().reset_index()
                cpu_sockets.columns = ['Socket', 'CPUs']
                
                mobo_sockets = datasets['motherboard']['socket'].value_counts().reset_index()
                mobo_sockets.columns = ['Socket', 'Motherboards']
                
                socket_df = pd.merge(cpu_sockets, mobo_sockets, on='Socket', how='outer').fillna(0)
                
                fig_sock = go.Figure()
                fig_sock.add_trace(go.Bar(
                    x=socket_df['Socket'], y=socket_df['CPUs'], name='CPUs',
                    marker_color='#3B82F6'
                ))
                fig_sock.add_trace(go.Bar(
                    x=socket_df['Socket'], y=socket_df['Motherboards'], name='Motherboards',
                    marker_color='#8B5CF6'
                ))
                fig_sock.update_layout(
                    barmode='group',
                    height=350,
                    xaxis_title="Socket Name",
                    yaxis_title="Count",
                    **plotly_layout_args
                )
                st.plotly_chart(fig_sock, width='stretch')
            
        with col_ov2:
            with st.container(key="glass-panel-insights-right"):
                st.markdown("##### Memory Standard Coexistence")
                st.write("RAM standard matching rules dictate motherboard selection:")
                
                # RAM types vs Mobo ram type counts
                ram_types = datasets['ram']['type'].value_counts().reset_index()
                ram_types.columns = ['RAM Type', 'RAM Kits']
                
                mobo_ram = datasets['motherboard']['ram_type'].value_counts().reset_index()
                mobo_ram.columns = ['RAM Type', 'Motherboards']
                
                ram_mobo_df = pd.merge(ram_types, mobo_ram, on='RAM Type', how='outer').fillna(0)
                
                fig_ram_mobo = go.Figure()
                fig_ram_mobo.add_trace(go.Bar(
                    x=ram_mobo_df['RAM Type'], y=ram_mobo_df['RAM Kits'], name='RAM Kits',
                    marker_color='#06B6D4'
                ))
                fig_ram_mobo.add_trace(go.Bar(
                    x=ram_mobo_df['RAM Type'], y=ram_mobo_df['Motherboards'], name='Motherboards',
                    marker_color='#3B82F6'
                ))
                fig_ram_mobo.update_layout(
                    barmode='group',
                    height=350,
                    xaxis_title="Memory Standard",
                    yaxis_title="Count",
                    **plotly_layout_args
                )
                st.plotly_chart(fig_ram_mobo, width='stretch')

    # ================= TAB 2: Price Analysis =================
    with tab_prices:
        st.subheader("Price Distributions Across Components")
        st.write("Understanding the price spread of components helps configure realistic budgets.")
        
        # Interactive selector for pricing distribution
        sel_comp = st.selectbox(
            "Select Component for Price Distribution:",
            options=display_names,
            index=0
        )
        
        comp_key = component_names[display_names.index(sel_comp)]
        df_selected = datasets[comp_key]
        
        fig_price = px.histogram(
            df_selected, x='price', 
            nbins=30, 
            color_discrete_sequence=['#3B82F6'],
            marginal="box",
            labels={'price': 'Price (USD)', 'count': 'Number of Components'},
            title=f"Price distribution of {sel_comp}"
        )
        fig_price.update_layout(
            height=400,
            **plotly_layout_args
        )
        st.plotly_chart(fig_price, width='stretch')
        
        # High level stats
        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        col_st1.metric("Min Price", f"${df_selected['price'].min():.2f}")
        col_st2.metric("Median Price", f"${df_selected['price'].median():.2f}")
        col_st3.metric("Average Price", f"${df_selected['price'].mean():.2f}")
        col_st4.metric("Max Price", f"${df_selected['price'].max():.2f}")

    # ================= TAB 3: CPU & Motherboard =================
    with tab_cpus_mobos:
        st.subheader("CPU Core Count & Clock Speed Analysis")
        
        col_cpu1, col_cpu2 = st.columns([1, 1], gap="medium")
        
        with col_cpu1:
            st.markdown("##### CPU Core Count Distribution")
            fig_cores = px.histogram(
                datasets['cpu'], x='cores',
                nbins=15, color_discrete_sequence=['#8B5CF6'],
                labels={'cores': 'Number of Cores', 'count': 'Count'}
            )
            fig_cores.update_layout(height=350, **plotly_layout_args)
            st.plotly_chart(fig_cores, width='stretch')
            
        with col_cpu2:
            st.markdown("##### Clock Speeds (Base vs Boost Clock)")
            fig_clock = px.scatter(
                datasets['cpu'], x='base_clock', y='boost_clock',
                color='cores', size='price', hover_name='name',
                color_continuous_scale=px.colors.sequential.Purples,
                labels={'base_clock': 'Base Clock (GHz)', 'boost_clock': 'Boost Clock (GHz)'}
            )
            fig_clock.update_layout(height=350, **plotly_layout_args)
            st.plotly_chart(fig_clock, width='stretch')

    # ================= TAB 4: GPU Performance =================
    with tab_gpus:
        st.subheader("GPU VRAM & Benchmark Distribution")
        
        col_gpu1, col_gpu2 = st.columns([1, 1], gap="medium")
        
        with col_gpu1:
            st.markdown("##### GPU VRAM Size (GB) Frequencies")
            fig_vram = px.histogram(
                datasets['gpu'], x='vram',
                nbins=15, color_discrete_sequence=['#06B6D4'],
                labels={'vram': 'VRAM Size (GB)', 'count': 'Count'}
            )
            fig_vram.update_layout(height=350, **plotly_layout_args)
            st.plotly_chart(fig_vram, width='stretch')
            
        with col_gpu2:
            st.markdown("##### Benchmark Scores vs Power Draw")
            fig_gpu_perf = px.scatter(
                datasets['gpu'], x='power_draw', y='benchmark_score',
                color='vram', size='price', hover_name='name',
                color_continuous_scale=px.colors.sequential.GnBu,
                labels={'power_draw': 'Power Draw (Watts)', 'benchmark_score': 'PassMark Benchmark Score'}
            )
            fig_gpu_perf.update_layout(height=350, **plotly_layout_args)
            st.plotly_chart(fig_gpu_perf, width='stretch')

    # ================= TAB 5: Memory & Storage =================
    with tab_ram_storage:
        st.subheader("RAM Speed & Storage Type Breakdown")
        
        col_mem1, col_mem2 = st.columns([1, 1], gap="medium")
        
        with col_mem1:
            st.markdown("##### RAM Speeds Distribution")
            fig_ram_speed = px.histogram(
                datasets['ram'], x='speed',
                nbins=15, color_discrete_sequence=['#3B82F6'],
                labels={'speed': 'RAM Speed (MHz)', 'count': 'Count'}
            )
            fig_ram_speed.update_layout(height=350, **plotly_layout_args)
            st.plotly_chart(fig_ram_speed, width='stretch')
            
        with col_mem2:
            st.markdown("##### Storage Drive Proportions by Type")
            # Proportions of SATA SSD, NVMe SSD, HDD, etc.
            store_types = datasets['storage']['type'].value_counts().reset_index()
            store_types.columns = ['Storage Type', 'Count']
            
            fig_store_pie = px.pie(
                store_types, values='Count', names='Storage Type',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_store_pie.update_layout(height=350, **plotly_layout_args)
            st.plotly_chart(fig_store_pie, width='stretch')

    # ================= TAB 6: PSU Efficiency =================
    with tab_psus:
        st.subheader("PSU Power Output & Efficiency Ratings")
        
        col_psu1, col_psu2 = st.columns([1, 1], gap="medium")
        
        with col_psu1:
            st.markdown("##### PSU Wattage Outputs")
            fig_watt = px.histogram(
                datasets['psu'], x='wattage',
                nbins=12, color_discrete_sequence=['#8B5CF6'],
                labels={'wattage': 'Wattage (Watts)', 'count': 'Count'}
            )
            fig_watt.update_layout(height=350, **plotly_layout_args)
            st.plotly_chart(fig_watt, width='stretch')
            
        with col_psu2:
            st.markdown("##### PSU Efficiency Rating Distribution")
            # Proportions of Bronze, Silver, Gold, Platinum, Titanium
            eff_counts = datasets['psu']['efficiency_rating'].value_counts().reset_index()
            eff_counts.columns = ['Efficiency Rating', 'Count']
            
            fig_eff = px.bar(
                eff_counts, x='Efficiency Rating', y='Count',
                color='Efficiency Rating',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_eff.update_layout(height=350, showlegend=False, **plotly_layout_args)
            st.plotly_chart(fig_eff, width='stretch')

if __name__ == "__main__":
    main()
