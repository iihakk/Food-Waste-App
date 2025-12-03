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

st.set_page_config(page_title="Food Waste Simulator", layout="wide")

st.title("Food Waste Reduction - Algorithm Testing Dashboard")

# sidebar controls
st.sidebar.header("Simulation Settings")

num_days = st.sidebar.slider("Number of Days", 1, 30, 7)
n_stores = st.sidebar.slider("Stores to Display (n)", 1, 10, 5)
shop_prob = st.sidebar.slider("Shopping Probability", 0.1, 1.0, 0.7)

st.sidebar.header("Algorithm Selection")
selected_algos = st.sidebar.multiselect(
    "Select algorithms to compare",
    options=list(ALGORITHMS.keys()),
    default=['Greedy Baseline']
)

st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use sample data", value=True)

if use_sample:
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    stores_path = os.path.join(data_dir, 'stores.csv')
    customers_path = os.path.join(data_dir, 'customers.csv')

    if not os.path.exists(stores_path):
        st.sidebar.info("Generating sample data...")
        generate_sample_data(data_dir)
else:
    stores_file = st.sidebar.file_uploader("Upload stores.csv", type='csv')
    customers_file = st.sidebar.file_uploader("Upload customers.csv", type='csv')

run_btn = st.sidebar.button("Run Simulation", type="primary")

# main area
if run_btn:
    if not selected_algos:
        st.error("Select at least one algorithm")
    else:
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
                st.error("Upload both CSV files")
                st.stop()

        st.subheader("Running simulation...")
        results = {}

        for algo_name in selected_algos:
            algo_func = ALGORITHMS[algo_name]
            engine = SimulationEngine(stores, customers)
            results[algo_name] = engine.run(num_days, n_stores, algo_func, shop_prob)

        st.success("Simulation complete!")

        # KPI comparison
        st.header("KPI Comparison")

        if len(selected_algos) == 1:
            algo = selected_algos[0]
            summary = results[algo]['summary']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bags Sold", summary['total_sold'])
            col2.metric("Bags Canceled", summary['total_canceled'])
            col3.metric("Bags Wasted", summary['total_wasted'])
            col4.metric("Revenue (EGP)", f"{summary['total_revenue']:,.2f}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Lost Revenue (EGP)", f"{summary['total_lost_revenue']:,.2f}")
            col6.metric("Fulfillment Rate", f"{summary['fulfillment_rate']}%")
            col7.metric("Waste Rate", f"{summary['waste_rate']}%")
            col8.metric("Customer Leave Rate", f"{summary['customer_leave_rate']}%")

        else:
            # comparison table
            comp_data = []
            for algo_name, res in results.items():
                s = res['summary']
                comp_data.append({
                    'Algorithm': algo_name,
                    'Bags Sold': s['total_sold'],
                    'Bags Canceled': s['total_canceled'],
                    'Bags Wasted': s['total_wasted'],
                    'Revenue (EGP)': s['total_revenue'],
                    'Lost Revenue (EGP)': s['total_lost_revenue'],
                    'Fulfillment %': s['fulfillment_rate'],
                    'Waste %': s['waste_rate'],
                    'Leave %': s['customer_leave_rate'],
                    'Fairness (std)': s['fairness_std']
                })

            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # charts
        st.header("Visualizations")

        tab1, tab2, tab3 = st.tabs(["Store Performance", "Daily Trends", "Fairness"])

        with tab1:
            # per-store breakdown for first algorithm
            algo = selected_algos[0]
            store_stats = results[algo]['store_stats']

            store_perf = []
            for sid, stat in store_stats.items():
                row = stores[stores['store_id'] == sid].iloc[0]
                store_perf.append({
                    'Store': f"{row['store_name']} ({row['branch']})",
                    'Sold': stat['sold'],
                    'Canceled': stat['canceled'],
                    'Wasted': stat['wasted'],
                    'Revenue': stat['revenue']
                })

            perf_df = pd.DataFrame(store_perf)

            fig = px.bar(perf_df, x='Store', y=['Sold', 'Canceled', 'Wasted'],
                         title=f"Store Performance - {algo}",
                         barmode='group')
            st.plotly_chart(fig, use_container_width=True)

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
                    'Bags Sold': day_sold,
                    'Bags Wasted': day_wasted,
                    'Revenue': day_rev
                })

            daily_df = pd.DataFrame(daily_summary)

            fig = px.line(daily_df, x='Day', y=['Bags Sold', 'Bags Wasted'],
                          title=f"Daily Performance - {algo}")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            algo = selected_algos[0]
            exposures = results[algo]['store_exposures']

            exp_data = []
            for sid, count in exposures.items():
                row = stores[stores['store_id'] == sid].iloc[0]
                exp_data.append({
                    'Store': f"{row['store_name']} ({row['branch']})",
                    'Times Shown': count
                })

            exp_df = pd.DataFrame(exp_data)
            fig = px.bar(exp_df, x='Store', y='Times Shown',
                         title=f"Store Exposure Distribution - {algo}")
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Exposure Standard Deviation (lower = fairer)",
                      results[algo]['summary']['fairness_std'])

        # detailed table
        st.header("Detailed Store Breakdown")

        algo = selected_algos[0]
        store_stats = results[algo]['store_stats']

        detail_rows = []
        for sid, stat in store_stats.items():
            row = stores[stores['store_id'] == sid].iloc[0]
            total_demand = stat['sold'] + stat['canceled']
            total_avail = stat['sold'] + stat['wasted']

            detail_rows.append({
                'Store ID': sid,
                'Name': row['store_name'],
                'Branch': row['branch'],
                'Rating': row['average_overall_rating'],
                'Sold': stat['sold'],
                'Canceled': stat['canceled'],
                'Wasted': stat['wasted'],
                'Revenue': round(stat['revenue'], 2),
                'Lost Revenue': round(stat['lost_revenue'], 2),
                'Fulfillment %': round(stat['sold'] / total_demand * 100, 1) if total_demand > 0 else 0,
                'Waste %': round(stat['wasted'] / total_avail * 100, 1) if total_avail > 0 else 0
            })

        detail_df = pd.DataFrame(detail_rows)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

else:
    st.info("Configure settings in the sidebar and click 'Run Simulation' to start.")

    # show data preview if available
    if use_sample:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        stores_path = os.path.join(data_dir, 'stores.csv')

        if os.path.exists(stores_path):
            st.subheader("Data Preview")
            stores, customers = load_data(stores_path,
                                          os.path.join(data_dir, 'customers.csv'))
            st.write("**Stores:**")
            st.dataframe(stores, hide_index=True)
            st.write(f"**Customers:** {len(customers)} records")
