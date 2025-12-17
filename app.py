"""
Food Waste Reduction Simulator Dashboard

A Streamlit dashboard for testing and comparing ranking algorithms
for a food waste reduction marketplace (surprise bag model).

Key Features:
- Compare multiple ranking algorithms
- Track KPIs: bags sold, items wasted, revenue, fairness
- Visualize store performance and daily trends
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.engine import SimulationEngine
from simulation.algorithms import ALGORITHMS, greedy_baseline, reset_genetic_algorithm
from simulation.data_loader import load_data, generate_sample_data

st.set_page_config(
    page_title="Food Waste Simulator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a4a6a;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #4361ee;
    }

    .stMetric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }

    .stMetric label {
        color: #4a4a6a !important;
        font-weight: 500;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #e9ecef;
    }

    section[data-testid="stSidebar"] label {
        color: #e9ecef !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #1a1a2e !important;
    }

    .stTabs [aria-selected="true"] {
        background: #4361ee !important;
    }

    .stTabs [aria-selected="true"] p {
        color: white !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e9ecef;
        border-radius: 10px;
    }
    
    /* Highlight best values in comparison table */
    .best-value {
        background-color: #d4edda;
        font-weight: bold;
    }
    
    .worst-value {
        background-color: #f8d7da;
    }
</style>
""", unsafe_allow_html=True)

# Color palette for charts
COLORS = ['#4361ee', '#3a0ca3', '#7209b7', '#f72585', '#4cc9f0']

# Header
st.markdown('<p class="main-header">Food Waste Reduction Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare ranking algorithms to minimize waste and maximize revenue</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### Simulation Settings")

    # Cloud mode for faster runs on Streamlit Cloud
    cloud_mode = st.checkbox(
        "Cloud Mode (faster)",
        value=False,
        help="Use smaller dataset and fewer days for faster runs on cloud hosting"
    )

    if cloud_mode:
        num_days = st.slider("Simulation Days", 1, 10, 3)
        st.caption("Limited to 3 days in cloud mode")
    else:
        num_days = st.slider("Simulation Days", 1, 30, 7)

    n_stores = st.slider("Stores Shown (n)", 1, 20, 10,
                         help="Number of stores displayed to each customer (realistic: 10-15)")
    shop_prob = st.slider("Shopping Probability", 0.1, 1.0, 0.7,
                          help="Chance a customer shops on any given day")

    st.markdown("---")
    st.markdown("### Business Model")

    midday_update = st.checkbox(
        "Mid-day Inventory Update",
        value=False,
        help="Stores report actual inventory at mid-day. Morning customers see estimates, afternoon customers see real remaining bags."
    )

    st.markdown("---")
    st.markdown("### Algorithm")

    selected_algos = st.multiselect(
        "Select to compare",
        options=list(ALGORITHMS.keys()),
        default=['Greedy Baseline']
    )

    st.markdown("---")
    randomize = st.checkbox("Randomize each run", value=False,
                           help="If unchecked, results are reproducible")

    # Show data generation sliders when randomize is enabled
    if randomize:
        st.markdown("---")
        st.markdown("### Random Data Generation")
        if cloud_mode:
            num_stores_gen = st.slider("Number of Stores", 10, 50, 20,
                                       help="Limited in cloud mode")
            num_customers_gen = st.slider("Number of Customers", 50, 500, 100,
                                          help="Limited in cloud mode")
        else:
            num_stores_gen = st.slider("Number of Stores", 10, 1000, 100,
                                       help="Number of stores to generate (supports up to 1000)")
            num_customers_gen = st.slider("Number of Customers", 100, 10000, 1000,
                                          help="Number of customers to generate (supports up to 10000)")
    else:
        num_stores_gen = 10
        num_customers_gen = 150

    st.markdown("---")
    st.markdown("### Data Source")

    use_sample = st.checkbox("Use sample data", value=True)

    if not use_sample:
        stores_file = st.file_uploader("stores.csv", type='csv')
        customers_file = st.file_uploader("customers.csv", type='csv')

    st.markdown("---")
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

# Main content
if run_btn:
    if not selected_algos:
        st.error("Please select at least one algorithm")
        st.stop()

    import time
    
    # Load data
    if use_sample:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # If randomize is enabled, generate fresh random data with custom sizes
        if randomize:
            run_seed = int(time.time() * 1000) % 100000
            with st.spinner(f"Generating {num_stores_gen} stores and {num_customers_gen} customers..."):
                stores, customers = generate_sample_data(
                    data_dir, 
                    seed=run_seed,
                    num_stores=num_stores_gen,
                    num_customers=num_customers_gen
                )
        else:
            stores_path = os.path.join(data_dir, 'stores.csv')
            customers_path = os.path.join(data_dir, 'customers.csv')
            if not os.path.exists(stores_path):
                generate_sample_data(data_dir)
            stores, customers = load_data(stores_path, customers_path)
    else:
        if stores_file and customers_file:
            stores = pd.read_csv(stores_file)
            customers = pd.read_csv(customers_file)
        else:
            st.error("Please upload both CSV files")
            st.stop()

    # Run simulation
    with st.spinner("Running simulation..."):
        results = {}
        execution_times = {}
        progress = st.progress(0)

        seed = None if randomize else 42

        for i, algo_name in enumerate(selected_algos):
            # Reset GA state before each algorithm run to ensure fair comparison
            reset_genetic_algorithm()
            
            algo_func = ALGORITHMS[algo_name]
            run_seed = int(time.time() * 1000) % 100000 if seed is None else seed
            engine = SimulationEngine(stores, customers, seed=run_seed)
            
            start_time = time.time()
            
            results[algo_name] = engine.run(num_days, n_stores, algo_func, shop_prob,
                                             midday_inventory_update=midday_update)
            
            end_time = time.time()
            execution_times[algo_name] = end_time - start_time
            
            progress.progress((i + 1) / len(selected_algos))

        progress.empty()

    st.success("Simulation complete!")

    # KPIs section
    st.markdown("## Key Performance Indicators")

    if len(selected_algos) == 1:
        algo = selected_algos[0]
        summary = results[algo]['summary']

        # Algorithm Score - the main comparison metric
        st.markdown("### Algorithm Score")
        score_col1, score_col2 = st.columns([1, 3])
        with score_col1:
            score = summary.get('algorithm_score', 0)
            if score >= 80:
                st.success(f"**{score:.1f}** / 100")
            elif score >= 60:
                st.warning(f"**{score:.1f}** / 100")
            else:
                st.error(f"**{score:.1f}** / 100")
        with score_col2:
            st.caption("Weighted: 30% Revenue + 30% Waste Reduction + 25% Satisfaction + 15% Fairness")

        st.markdown("### Core Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Reservations", f"{summary.get('total_reservations', 0):,}")
        with col2:
            st.metric("Fulfilled", f"{summary.get('total_fulfilled', 0):,}")
        with col3:
            st.metric("Cancelled", f"{summary.get('total_cancelled', 0):,}")
        with col4:
            st.metric("Unsold (Waste)", f"{summary.get('total_unsold', 0):,}")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Revenue", f"{summary['total_revenue']:,.0f} EGP")
        with col6:
            st.metric("Lost Revenue", f"{summary.get('total_lost_revenue', 0):,.0f} EGP")
        with col7:
            st.metric("Revenue Efficiency", f"{summary['revenue_efficiency']}%")
        with col8:
            st.metric("Waste Rate", f"{summary['waste_rate']}%")

        col9, col10, col11, col12 = st.columns(4)
        with col9:
            st.metric("Demand Fulfillment", f"{summary.get('demand_fulfillment', 0):.1f}%")
        with col10:
            st.metric("Cancellation Rate", f"{summary.get('cancellation_rate', 0):.1f}%")
        with col11:
            st.metric("Leave Rate", f"{summary['customer_leave_rate']}%")
        with col12:
            st.metric("Fairness Score", f"{summary.get('fairness_score', 0):.1f}/100")

    else:
        # Comparison table for multiple algorithms - Option 2: Dual View
        comp_data = []
        for algo_name, res in results.items():
            s = res['summary']

            # Get revenue metrics
            revenue = s.get('total_revenue', 0)
            lost_revenue = s.get('total_lost_revenue', 0)  # This is unsold × price

            # Get bag counts
            unsold = s.get('total_unsold', 0)

            comp_data.append({
                'Algorithm': algo_name,
                'Revenue': revenue,
                'Waste (bags)': unsold,
                'Waste (EGP)': lost_revenue,
                'Runtime': f"{execution_times[algo_name]:.3f}s"
            })

        comp_df = pd.DataFrame(comp_data)

        # Sort by Revenue descending (best first)
        comp_df = comp_df.sort_values('Revenue', ascending=False)

        # Format for display
        display_df = comp_df.copy()
        display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"{x:,.0f}")
        display_df['Waste (EGP)'] = display_df['Waste (EGP)'].apply(lambda x: f"{x:,.0f}")

        # Display the comparison table
        st.markdown("### Algorithm Comparison")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # ============================================
        # Overall Comparison Chart (Grouped Bar)
        # ============================================
        st.markdown("### Overall Algorithm Comparison")

        # Create grouped bar chart comparing Revenue vs Waste (EGP)
        comparison_melted = comp_df.melt(
            id_vars=['Algorithm'],
            value_vars=['Revenue', 'Waste (EGP)'],
            var_name='Metric',
            value_name='Value'
        )

        fig_comparison = px.bar(
            comparison_melted,
            x='Algorithm',
            y='Value',
            color='Metric',
            barmode='group',
            title='Revenue vs Waste by Algorithm',
            color_discrete_sequence=[COLORS[0], COLORS[3]]
        )
        fig_comparison.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis_title='Amount (EGP)'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        # ============================================
        # Revenue & Waste Charts
        # ============================================
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Bar chart: Revenue per algorithm
            fig_rev = px.bar(
                comp_df,
                x='Algorithm',
                y='Revenue',
                title='Revenue by Algorithm',
                color='Revenue',
                color_continuous_scale='Greens'
            )
            fig_rev.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                coloraxis_showscale=False,
                yaxis_title='Revenue (EGP)'
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with chart_col2:
            # Bar chart: Waste (EGP) per algorithm
            fig_waste_egp = px.bar(
                comp_df,
                x='Algorithm',
                y='Waste (EGP)',
                title='Waste (EGP) by Algorithm',
                color='Waste (EGP)',
                color_continuous_scale='Reds'
            )
            fig_waste_egp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                coloraxis_showscale=False,
                yaxis_title='Waste (EGP)'
            )
            st.plotly_chart(fig_waste_egp, use_container_width=True)

        # ============================================
        # Waste (Bags) Chart
        # ============================================
        fig_waste_bags = px.bar(
            comp_df,
            x='Algorithm',
            y='Waste (bags)',
            title='Waste (Bags) by Algorithm',
            color='Waste (bags)',
            color_continuous_scale='Oranges'
        )
        fig_waste_bags.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False,
            yaxis_title='Unsold Bags'
        )
        st.plotly_chart(fig_waste_bags, use_container_width=True)

        # ============================================
        # Performance Highlights
        # ============================================
        st.markdown("### Performance Highlights")

        highlight_col1, highlight_col2, highlight_col3 = st.columns(3)

        with highlight_col1:
            # Highest revenue
            best_rev_idx = comp_df['Revenue'].idxmax()
            best_rev_algo = comp_df.loc[best_rev_idx, 'Algorithm']
            best_rev = comp_df.loc[best_rev_idx, 'Revenue']
            st.success(f"**Highest Revenue**\n\n{best_rev_algo}\n\n{best_rev:,.0f} EGP")

        with highlight_col2:
            # Lowest waste (EGP)
            best_waste_egp_idx = comp_df['Waste (EGP)'].idxmin()
            best_waste_egp_algo = comp_df.loc[best_waste_egp_idx, 'Algorithm']
            best_waste_egp = comp_df.loc[best_waste_egp_idx, 'Waste (EGP)']
            st.success(f"**Lowest Waste (EGP)**\n\n{best_waste_egp_algo}\n\n{best_waste_egp:,.0f} EGP")

        with highlight_col3:
            # Lowest waste (bags)
            best_waste_bags_idx = comp_df['Waste (bags)'].idxmin()
            best_waste_bags_algo = comp_df.loc[best_waste_bags_idx, 'Algorithm']
            best_waste_bags = comp_df.loc[best_waste_bags_idx, 'Waste (bags)']
            st.success(f"**Lowest Waste (Bags)**\n\n{best_waste_bags_algo}\n\n{best_waste_bags:,} bags")

    # Charts
    st.markdown("## Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Store Performance", "Daily Trends", "Fairness", "Waste Analysis"])

    with tab1:
        algo = selected_algos[0]
        store_stats = results[algo]['store_stats']

        store_perf = []
        for sid, stat in store_stats.items():
            row = stores[stores['store_id'] == sid].iloc[0]
            store_perf.append({
                'Store': f"{row['store_name']} ({row['branch']})",
                'Fulfilled': stat['fulfilled'],
                'Cancelled': stat['cancelled'],
                'Unsold': stat['unsold']
            })

        perf_df = pd.DataFrame(store_perf)

        fig = px.bar(perf_df, x='Store', y=['Fulfilled', 'Cancelled', 'Unsold'],
                     barmode='group', color_discrete_sequence=[COLORS[0], COLORS[2], COLORS[3]])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=40),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        algo = selected_algos[0]
        daily = results[algo]['daily_data']

        daily_summary = []
        for d in daily:
            day_fulfilled = sum(s['fulfilled'] for s in d['stores'].values())
            day_cancelled = sum(s['cancelled'] for s in d['stores'].values())
            day_unsold = sum(s['unsold'] for s in d['stores'].values())
            day_rev = sum(s['revenue'] for s in d['stores'].values())
            daily_summary.append({
                'Day': d['day'],
                'Fulfilled': day_fulfilled,
                'Cancelled': day_cancelled,
                'Unsold': day_unsold,
                'Revenue': day_rev
            })

        daily_df = pd.DataFrame(daily_summary)

        fig = px.line(daily_df, x='Day', y=['Fulfilled', 'Cancelled', 'Unsold'],
                      color_discrete_sequence=[COLORS[0], COLORS[2], COLORS[3]],
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        algo = selected_algos[0]
        exposures = results[algo]['store_exposures']

        exp_data = []
        for sid, count in exposures.items():
            row = stores[stores['store_id'] == sid].iloc[0]
            exp_data.append({
                'Store': f"{row['store_name']} ({row['branch']})",
                'Exposures': count
            })

        exp_df = pd.DataFrame(exp_data).sort_values('Exposures', ascending=True)

        fig = px.bar(exp_df, x='Exposures', y='Store', orientation='h',
                     color='Exposures', color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False,
            height=max(400, len(exp_df) * 25)
        )
        st.plotly_chart(fig, use_container_width=True)

        fairness = results[algo]['summary']['fairness_std']
        if fairness < 30:
            st.success(f"Fairness Score: {fairness:.1f} (Good - stores get similar exposure)")
        elif fairness < 60:
            st.warning(f"Fairness Score: {fairness:.1f} (Moderate - some stores favored)")
        else:
            st.error(f"Fairness Score: {fairness:.1f} (Poor - significant imbalance)")

    # ============================================
    # NEW TAB: Waste Analysis
    # ============================================
    with tab4:
        st.markdown("### Waste Analysis by Store")
        
        algo = selected_algos[0]
        store_stats = results[algo]['store_stats']
        summary = results[algo]['summary']
        
        # Calculate waste metrics per store
        waste_data = []
        for sid, stat in store_stats.items():
            row = stores[stores['store_id'] == sid].iloc[0]
            total_available = stat['fulfilled'] + stat['unsold'] + stat['cancelled']
            waste_pct = (stat['unsold'] / total_available * 100) if total_available > 0 else 0
            
            waste_data.append({
                'Store': f"{row['store_name']} ({row['branch']})",
                'Store ID': sid,
                'Total Available': total_available,
                'Fulfilled': stat['fulfilled'],
                'Unsold (Waste)': stat['unsold'],
                'Waste %': waste_pct,
                'Revenue Lost': stat['unsold'] * row['price']  # Assuming price column exists
            })
        
        waste_df = pd.DataFrame(waste_data).sort_values('Waste %', ascending=False)
        
        # Display waste summary
        total_unsold = summary.get('total_unsold', 0)
        total_fulfilled = summary.get('total_fulfilled', 0)
        waste_rate = summary.get('waste_rate', 0)
        
        waste_sum_col1, waste_sum_col2, waste_sum_col3, waste_sum_col4 = st.columns(4)
        
        with waste_sum_col1:
            st.metric("Total Bags Wasted", f"{total_unsold:,}")
        with waste_sum_col2:
            st.metric("Total Bags Sold", f"{total_fulfilled:,}")
        with waste_sum_col3:
            st.metric("Overall Waste Rate", f"{waste_rate}%")
        with waste_sum_col4:
            waste_reduction = 100 - float(waste_rate)
            st.metric("Waste Reduction Score", f"{waste_reduction:.1f}%")
        
        st.markdown("---")
        
        # Waste per store chart
        waste_chart_col1, waste_chart_col2 = st.columns(2)
        
        with waste_chart_col1:
            # Top wasters
            st.markdown("#### Stores with Highest Waste")
            top_wasters = waste_df.head(10)
            
            fig_top_waste = px.bar(
                top_wasters,
                x='Store',
                y='Unsold (Waste)',
                color='Waste %',
                color_continuous_scale='Reds',
                title='Top 10 Stores by Waste Volume'
            )
            fig_top_waste.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_top_waste, use_container_width=True)
        
        with waste_chart_col2:
            # Waste rate distribution
            st.markdown("#### Waste Rate Distribution")
            
            fig_waste_dist = px.histogram(
                waste_df,
                x='Waste %',
                nbins=20,
                title='Distribution of Waste Rates Across Stores',
                color_discrete_sequence=[COLORS[3]]
            )
            fig_waste_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_waste_dist, use_container_width=True)
        
        # Waste by Store bar chart (replaced treemap due to library compatibility)
        st.markdown("#### Waste by Store")

        waste_by_store = waste_df[waste_df['Unsold (Waste)'] > 0].sort_values('Unsold (Waste)', ascending=True)
        if len(waste_by_store) > 0:
            fig_waste_bar = px.bar(
                waste_by_store,
                x='Unsold (Waste)',
                y='Store',
                orientation='h',
                color='Waste %',
                color_continuous_scale='RdYlGn_r',
                title='Waste by Store (Color = Waste %)'
            )
            fig_waste_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=max(400, len(waste_by_store) * 25)
            )
            st.plotly_chart(fig_waste_bar, use_container_width=True)
        else:
            st.info("No waste recorded - all bags were sold!")
        
        # Detailed table
        st.markdown("#### Detailed Waste Breakdown")
        st.dataframe(
            waste_df[['Store', 'Total Available', 'Fulfilled', 'Unsold (Waste)', 'Waste %']].round(2),
            use_container_width=True,
            hide_index=True
        )
        
        # Waste insights
        st.markdown("#### Insights")
        
        # Find stores with zero waste
        zero_waste_stores = waste_df[waste_df['Unsold (Waste)'] == 0]
        high_waste_stores = waste_df[waste_df['Waste %'] > 50]
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            if len(zero_waste_stores) > 0:
                st.success(f"✅ **{len(zero_waste_stores)} stores** achieved zero waste!")
            else:
                st.warning("⚠️ No stores achieved zero waste in this simulation.")
        
        with insight_col2:
            if len(high_waste_stores) > 0:
                st.error(f"❌ **{len(high_waste_stores)} stores** have >50% waste rate and need attention.")
            else:
                st.success("✅ No stores have critically high waste rates (>50%).")

    # Detailed breakdown
    st.markdown("## Store Details")

    algo = selected_algos[0]
    store_stats = results[algo]['store_stats']

    detail_rows = []
    for sid, stat in store_stats.items():
        row = stores[stores['store_id'] == sid].iloc[0]
        
        # Calculate waste rate per store
        total_available = stat['fulfilled'] + stat['unsold'] + stat['cancelled']
        waste_pct = (stat['unsold'] / total_available * 100) if total_available > 0 else 0

        detail_rows.append({
            'ID': sid,
            'Store': row['store_name'],
            'Branch': row['branch'],
            'Rating': row['average_overall_rating'],
            'Reservations': stat['reservations'],
            'Fulfilled': stat['fulfilled'],
            'Cancelled': stat['cancelled'],
            'Unsold (Waste)': stat['unsold'],  # <-- Renamed for clarity
            'Waste %': f"{waste_pct:.1f}%",  # <-- NEW: Waste percentage column
            'Revenue': f"{stat['revenue']:,.0f}",
        })

    detail_df = pd.DataFrame(detail_rows)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

else:
    # Welcome state
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### How it works

        1. **Configure** simulation parameters in the sidebar
        2. **Select** one or more ranking algorithms to compare
        3. **Run** the simulation to see results

        **Surprise Bag Model:**
        - Each store has X items worth of food
        - Customers buy 1 bag each at fixed price
        - If N customers order, each bag contains X/N items
        - If 0 customers order from a store, those items are wasted
        """)

    with col2:
        st.markdown("""
        ### Metrics tracked

        - Bags sold (customers served)
        - Items wasted vs distributed
        - Avg items per bag (customer value)
        - Revenue & efficiency
        - Store exposure fairness
        - **Waste rate & reduction**
        """)

    # Data preview
    if use_sample:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        stores_path = os.path.join(data_dir, 'stores.csv')

        if os.path.exists(stores_path):
            st.markdown("---")
            st.markdown("### Data Preview")

            stores, customers = load_data(stores_path, os.path.join(data_dir, 'customers.csv'))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{len(stores)} Stores**")
                st.dataframe(stores[['store_name', 'branch', 'average_overall_rating', 'price']],
                            hide_index=True, height=300)
            with col2:
                st.markdown(f"**{len(customers)} Customers**")
                st.dataframe(customers.head(10), hide_index=True, height=300)