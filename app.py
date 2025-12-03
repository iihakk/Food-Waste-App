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

# custom styling
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

# color palette for charts
COLORS = ['#4361ee', '#3a0ca3', '#7209b7', '#f72585', '#4cc9f0']

# header
st.markdown('<p class="main-header">Food Waste Reduction Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare ranking algorithms to minimize waste and maximize revenue</p>', unsafe_allow_html=True)

# sidebar
with st.sidebar:
    st.markdown("### Simulation Settings")

    num_days = st.slider("Simulation Days", 1, 30, 7)
    n_stores = st.slider("Stores Shown (n)", 1, 10, 5,
                         help="Number of stores displayed to each customer")
    shop_prob = st.slider("Shopping Probability", 0.1, 1.0, 0.7,
                          help="Chance a customer shops on any given day")

    st.markdown("---")
    st.markdown("### Algorithm")

    selected_algos = st.multiselect(
        "Select to compare",
        options=list(ALGORITHMS.keys()),
        default=['Greedy Baseline']
    )

    st.markdown("---")
    randomize = st.checkbox("Randomize each run", value=False,
                           help="If unchecked, results are reproducible (same every time)")

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
    run_btn = st.button("Run Simulation", type="primary", width="stretch")

# main content
if run_btn:
    if not selected_algos:
        st.error("Please select at least one algorithm")
        st.stop()

    # load data
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

    # run simulation
    with st.spinner("Running simulation..."):
        results = {}
        progress = st.progress(0)

        import time
        seed = None if randomize else 42

        for i, algo_name in enumerate(selected_algos):
            algo_func = ALGORITHMS[algo_name]
            # use timestamp as seed if randomizing
            run_seed = int(time.time() * 1000) % 100000 if seed is None else seed
            engine = SimulationEngine(stores, customers, seed=run_seed)
            results[algo_name] = engine.run(num_days, n_stores, algo_func, shop_prob)
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
            st.metric("Bags Sold", f"{summary['total_sold']:,}")
        with col2:
            st.metric("Bags Canceled", f"{summary['total_canceled']:,}")
        with col3:
            st.metric("Bags Wasted", f"{summary['total_wasted']:,}")
        with col4:
            st.metric("Revenue", f"{summary['total_revenue']:,.0f} EGP")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Lost Revenue", f"{summary['total_lost_revenue']:,.0f} EGP")
        with col6:
            st.metric("Fulfillment Rate", f"{summary['fulfillment_rate']}%")
        with col7:
            st.metric("Waste Rate", f"{summary['waste_rate']}%")
        with col8:
            st.metric("Leave Rate", f"{summary['customer_leave_rate']}%")

    else:
        comp_data = []
        for algo_name, res in results.items():
            s = res['summary']
            comp_data.append({
                'Algorithm': algo_name,
                'Sold': s['total_sold'],
                'Canceled': s['total_canceled'],
                'Wasted': s['total_wasted'],
                'Revenue': f"{s['total_revenue']:,.0f}",
                'Lost': f"{s['total_lost_revenue']:,.0f}",
                'Fulfill %': s['fulfillment_rate'],
                'Waste %': s['waste_rate'],
                'Leave %': s['customer_leave_rate'],
                'Fairness': s['fairness_std']
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, width="stretch", hide_index=True)

    # charts
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
                'Sold': stat['sold'],
                'Canceled': stat['canceled'],
                'Wasted': stat['wasted']
            })

        perf_df = pd.DataFrame(store_perf)

        fig = px.bar(perf_df, x='Store', y=['Sold', 'Canceled', 'Wasted'],
                     barmode='group', color_discrete_sequence=COLORS[:3])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=40)
        )
        st.plotly_chart(fig, width="stretch")

    with tab2:
        algo = selected_algos[0]
        daily = results[algo]['daily_data']

        daily_summary = []
        for d in daily:
            day_sold = sum(s['sold'] for s in d['stores'].values())
            day_wasted = sum(s['wasted'] for s in d['stores'].values())
            day_rev = sum(s['revenue'] for s in d['stores'].values())
            daily_summary.append({
                'Day': d['day'],
                'Sold': day_sold,
                'Wasted': day_wasted,
                'Revenue': day_rev
            })

        daily_df = pd.DataFrame(daily_summary)

        fig = px.line(daily_df, x='Day', y=['Sold', 'Wasted'],
                      color_discrete_sequence=[COLORS[0], COLORS[3]],
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, width="stretch")

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
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, width="stretch")

        fairness = results[algo]['summary']['fairness_std']
        if fairness < 30:
            st.success(f"Fairness Score: {fairness:.1f} (Good - stores get similar exposure)")
        elif fairness < 60:
            st.warning(f"Fairness Score: {fairness:.1f} (Moderate - some stores favored)")
        else:
            st.error(f"Fairness Score: {fairness:.1f} (Poor - significant imbalance)")

    # detailed breakdown
    st.markdown("## Store Details")

    algo = selected_algos[0]
    store_stats = results[algo]['store_stats']

    detail_rows = []
    for sid, stat in store_stats.items():
        row = stores[stores['store_id'] == sid].iloc[0]
        total_demand = stat['sold'] + stat['canceled']
        total_avail = stat['sold'] + stat['wasted']

        detail_rows.append({
            'ID': sid,
            'Store': row['store_name'],
            'Branch': row['branch'],
            'Rating': row['average_overall_rating'],
            'Sold': stat['sold'],
            'Canceled': stat['canceled'],
            'Wasted': stat['wasted'],
            'Revenue': f"{stat['revenue']:,.0f}",
            'Lost': f"{stat['lost_revenue']:,.0f}",
            'Fulfill %': round(stat['sold'] / total_demand * 100, 1) if total_demand > 0 else 0,
            'Waste %': round(stat['wasted'] / total_avail * 100, 1) if total_avail > 0 else 0
        })

    detail_df = pd.DataFrame(detail_rows)
    st.dataframe(detail_df, width="stretch", hide_index=True)

else:
    # welcome state
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### How it works

        1. **Configure** simulation parameters in the sidebar
        2. **Select** one or more ranking algorithms to compare
        3. **Run** the simulation to see results

        The simulator models customer behavior over multiple days, tracking how different
        store ranking strategies affect waste, revenue, and fairness.
        """)

    with col2:
        st.markdown("""
        ### Metrics tracked

        - Bags sold vs wasted
        - Revenue generated & lost
        - Customer fulfillment rate
        - Store exposure fairness
        """)

    # data preview
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
