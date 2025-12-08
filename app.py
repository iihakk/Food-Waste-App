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
from simulation.algorithms import ALGORITHMS, greedy_baseline
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

    num_days = st.slider("Simulation Days", 1, 30, 7)
    n_stores = st.slider("Stores Shown (n)", 1, 10, 5,
                         help="Number of stores displayed to each customer")
    shop_prob = st.slider("Shopping Probability", 0.1, 1.0, 0.7,
                          help="Chance a customer shops on any given day")

    st.markdown("---")
    st.markdown("### Advanced Settings")

    use_accuracy = st.checkbox("Use Accuracy Adjustment", value=True,
                               help="Adjust estimates based on store history")
    alt_accept_rate = st.slider("Alternative Acceptance Rate", 0.0, 1.0, 0.6,
                                help="Chance customer accepts alternative when cancelled")

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

    st.markdown("---")
    st.markdown("### Data Source")

    use_sample = st.checkbox("Use sample data", value=True)

    if use_sample:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        stores_path = os.path.join(data_dir, 'stores.csv')
        customers_path = os.path.join(data_dir, 'customers.csv')
        if not os.path.exists(stores_path):
            generate_sample_data(data_dir)
    else:
        stores_file = st.file_uploader("stores.csv", type='csv')
        customers_file = st.file_uploader("customers.csv", type='csv')

    st.markdown("---")
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

# Main content
if run_btn:
    if not selected_algos:
        st.error("Please select at least one algorithm")
        st.stop()

    # Load data
    if use_sample:
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
        progress = st.progress(0)

        import time
        seed = None if randomize else 42

        for i, algo_name in enumerate(selected_algos):
            algo_func = ALGORITHMS[algo_name]
            run_seed = int(time.time() * 1000) % 100000 if seed is None else seed
            engine = SimulationEngine(stores, customers, seed=run_seed)
            results[algo_name] = engine.run(num_days, n_stores, algo_func, shop_prob,
                                           use_accuracy_adjustment=use_accuracy,
                                           alternative_acceptance_rate=alt_accept_rate)
            progress.progress((i + 1) / len(selected_algos))

        progress.empty()

    st.success("Simulation complete!")

    # KPIs section
    st.markdown("## Key Performance Indicators")

    if len(selected_algos) == 1:
        algo = selected_algos[0]
        summary = results[algo]['summary']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Bags Sold", f"{summary['total_bags_sold']:,}")
        with col2:
            st.metric("Items Wasted", f"{summary['total_items_wasted']:,}")
        with col3:
            st.metric("Avg Items/Bag", f"{summary['avg_items_per_bag']:.1f}")
        with col4:
            st.metric("Revenue", f"{summary['total_revenue']:,.0f} EGP")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Revenue Efficiency", f"{summary['revenue_efficiency']}%")
        with col6:
            st.metric("Waste Rate", f"{summary['waste_rate']}%")
        with col7:
            st.metric("Leave Rate", f"{summary['customer_leave_rate']}%")
        with col8:
            st.metric("Fairness (std)", f"{summary['fairness_std']:.1f}")

        # New KPIs row
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            st.metric("Cancellations", f"{summary.get('total_cancellations', 0):,}")
        with col10:
            st.metric("Cancellation Rate", f"{summary.get('cancellation_rate', 0):.1f}%")
        with col11:
            st.metric("Avg Accuracy", f"{summary.get('avg_store_accuracy', 1.0):.2f}")
        with col12:
            st.metric("Satisfaction", f"{summary.get('customer_satisfaction_score', 0):.0f}/100")

    else:
        # Comparison table for multiple algorithms
        comp_data = []
        for algo_name, res in results.items():
            s = res['summary']
            comp_data.append({
                'Algorithm': algo_name,
                'Bags Sold': s['total_bags_sold'],
                'Items Wasted': s['total_items_wasted'],
                'Revenue': f"{s['total_revenue']:,.0f}",
                'Waste %': s['waste_rate'],
                'Cancel %': s.get('cancellation_rate', 0),
                'Leave %': s['customer_leave_rate'],
                'Satisfaction': s.get('customer_satisfaction_score', 0),
                'Fairness': s['fairness_std']
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Charts
    st.markdown("## Analysis")

    tab1, tab2, tab3 = st.tabs(["Store Performance", "Daily Trends", "Fairness"])

    with tab1:
        algo = selected_algos[0]
        store_stats = results[algo]['store_stats']

        store_perf = []
        for sid, stat in store_stats.items():
            row = stores[stores['store_id'] == sid].iloc[0]
            store_perf.append({
                'Store': f"{row['store_name']} ({row['branch']})",
                'Bags Sold': stat['bags_sold'],
                'Items Distributed': stat['items_distributed'],
                'Items Wasted': stat['items_wasted']
            })

        perf_df = pd.DataFrame(store_perf)

        fig = px.bar(perf_df, x='Store', y=['Bags Sold', 'Items Wasted'],
                     barmode='group', color_discrete_sequence=[COLORS[0], COLORS[3]])
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
            day_bags = sum(s['bags_sold'] for s in d['stores'].values())
            day_wasted = sum(s['items_wasted'] for s in d['stores'].values())
            day_rev = sum(s['revenue'] for s in d['stores'].values())
            daily_summary.append({
                'Day': d['day'],
                'Bags Sold': day_bags,
                'Items Wasted': day_wasted,
                'Revenue': day_rev
            })

        daily_df = pd.DataFrame(daily_summary)

        fig = px.line(daily_df, x='Day', y=['Bags Sold', 'Items Wasted'],
                      color_discrete_sequence=[COLORS[0], COLORS[3]],
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

    # Detailed breakdown
    st.markdown("## Store Details")

    algo = selected_algos[0]
    store_stats = results[algo]['store_stats']

    detail_rows = []
    for sid, stat in store_stats.items():
        row = stores[stores['store_id'] == sid].iloc[0]

        detail_rows.append({
            'ID': sid,
            'Store': row['store_name'],
            'Branch': row['branch'],
            'Rating': row['average_overall_rating'],
            'Bags Sold': stat['bags_sold'],
            'Items Wasted': stat['items_wasted'],
            'Cancellations': stat.get('cancellations', 0),
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
