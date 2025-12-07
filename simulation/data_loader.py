"""
Data Loader for Food Waste Reduction Platform

This module handles loading and generating data for the simulation.

Data Format:
------------

stores.csv columns:
    - store_id: Unique integer identifier
    - store_name: Store/bakery name (e.g., "La Poire", "El Abd")
    - branch: Location/area (e.g., "Zamalek", "Maadi")
    - average_bags_at_9AM: Estimated bags available at start of day
    - average_overall_rating: Customer rating 1-5 (float)
    - price: Bag price in EGP
    - longitude: Store longitude (Cairo coordinates)
    - latitude: Store latitude (Cairo coordinates)

customers.csv columns:
    - customer_id: Unique integer identifier
    - longitude: Customer location longitude
    - latitude: Customer location latitude
    - store{N}_valuation: Customer's preference for store N (1-5)
      One column per store, e.g., store100_valuation, store101_valuation, etc.

Cairo Area Coordinates Reference:
    - Zamalek: ~31.22, 30.06
    - Maadi: ~31.26, 29.96
    - Heliopolis: ~31.34, 30.09
    - New Cairo: ~31.47, 30.02
    - Mohandessin: ~31.20, 30.05
    - Downtown: ~31.24, 30.05
"""

import pandas as pd
import numpy as np
import random
import os


def load_data(stores_path, customers_path):
    """
    Load store and customer data from CSV files.

    Args:
        stores_path (str): Path to stores.csv
        customers_path (str): Path to customers.csv

    Returns:
        tuple: (stores_df, customers_df) DataFrames
    """
    stores = pd.read_csv(stores_path)
    customers = pd.read_csv(customers_path)
    return stores, customers


def generate_sample_data(output_dir='data', seed=44):
    """
    Generate sample store and customer data for testing.

    This function creates realistic sample data with:
    - 10 stores across Cairo neighborhoods
    - 150 customers with random locations and valuations

    Note: This is for initial testing only. For the actual simulation,
    use the manually curated stores.csv with Egyptian bakeries.

    Args:
        output_dir (str): Directory to save CSV files
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (stores_df, customers_df) DataFrames
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Sample stores data (original 10 stores)
    stores = pd.DataFrame({
        'store_id': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'store_name': ['TBS', 'Dunkin', 'Costa Coffee', 'Starbucks', 'Paul',
                       'TBS', 'Cilantro', "Beano's", 'Harris Cafe', 'Dunkin'],
        'branch': ['Zamalek', 'New Cairo', 'Maadi', 'Heliopolis', 'Mohandessin',
                   'New Cairo', 'Zamalek', 'Maadi', 'Heliopolis', 'Mohandessin'],
        'average_bags_at_9AM': [random.randint(5, 30) for _ in range(10)],
        'average_overall_rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(10)],
        'price': [round(random.uniform(25.0, 75.0), 2) for _ in range(10)],
        # Cairo coordinates for each area
        'longitude': [31.2194, 31.4913, 31.2625, 31.3375, 31.2001,
                      31.4701, 31.2243, 31.2587, 31.3421, 31.2056],
        'latitude': [30.0626, 30.0171, 29.9602, 30.0911, 30.0561,
                     30.0285, 30.0651, 29.9711, 30.0875, 30.0512]
    })

    # Generate customer data
    num_customers = 150
    customers = pd.DataFrame({
        'customer_id': list(range(1, num_customers + 1)),
        # Random locations within Cairo area
        'longitude': [round(31.2 + np.random.uniform(-0.1, 0.3), 4) for _ in range(num_customers)],
        'latitude': [round(30.0 + np.random.uniform(-0.1, 0.1), 4) for _ in range(num_customers)]
    })

    # Add valuation column for each store
    # Valuation represents how much a customer prefers that store (1-5)
    for sid in stores['store_id']:
        customers[f'store{sid}_valuation'] = [random.randint(1, 5) for _ in range(num_customers)]

    # Save to CSV
    stores.to_csv(os.path.join(output_dir, 'stores.csv'), index=False)
    customers.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)

    return stores, customers
