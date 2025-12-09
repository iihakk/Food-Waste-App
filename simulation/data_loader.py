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


def generate_sample_data(output_dir='data', num_stores=200, num_customers=500, seed=None):
    """
    Generate random store and customer data for simulation.

    This function creates randomized data with:
    - Configurable number of stores across Cairo neighborhoods
    - Configurable number of customers with random locations and valuations
    - Fresh random data on each run (no fixed seed by default)

    Args:
        output_dir (str): Directory to save CSV files
        num_stores (int): Number of stores to generate (default: 200)
        num_customers (int): Number of customers to generate (default: 500)
        seed (int, optional): Random seed for reproducibility. If None, generates fresh random data.

    Returns:
        tuple: (stores_df, customers_df) DataFrames
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Store name options (Egyptian bakeries and cafes)
    store_names = [
        'TBS', 'Dunkin', 'Costa Coffee', 'Starbucks', 'Paul', 'Cilantro',
        "Beano's", 'Harris Cafe', 'La Poire', 'El Abd', 'Mandarine Koueider',
        'Etoile', 'Simonds', 'Dukes', 'Nola Cupcakes', 'Patchi', 'Cinnabon',
        'Krispy Kreme', 'Breadwinner', 'Lychee', 'Cake Maison', 'Dipndip',
        'Auntie Anne\'s', 'Marble Slab', 'Baskin Robbins', 'Cold Stone',
        'Tutti Frutti', 'Cookiedoodle', 'House of Donuts', 'Bake Rolz',
        'Bisco Misr', 'Edita', 'Fresh Food', 'Gourmet', 'Le Chantilly'
    ]

    # Cairo neighborhoods with bounding box coordinates
    # Format: 'Neighborhood': (lon_min, lon_max, lat_min, lat_max)
    neighborhoods = {
        'Zamalek': (31.21, 31.23, 30.05, 30.07),
        'Maadi': (31.24, 31.28, 29.94, 29.98),
        'Heliopolis': (31.32, 31.36, 30.08, 30.11),
        'New Cairo': (31.42, 31.52, 29.99, 30.05),
        'Mohandessin': (31.19, 31.21, 30.04, 30.07),
        'Downtown': (31.23, 31.26, 30.04, 30.06),
        'Nasr City': (31.31, 31.36, 30.05, 30.08),
        'Dokki': (31.20, 31.22, 30.02, 30.05),
        'Giza': (31.19, 31.22, 29.99, 30.02),
        '6th October': (30.88, 30.98, 29.92, 29.98),
        'Sheikh Zayed': (30.92, 30.98, 30.00, 30.04),
        'Rehab': (31.48, 31.52, 30.04, 30.08),
        'Tagamoa': (31.42, 31.48, 29.99, 30.03),
        'Korba': (31.32, 31.34, 30.08, 30.10),
        'Shoubra': (31.23, 31.26, 30.07, 30.10)
    }

    neighborhood_names = list(neighborhoods.keys())

    # Generate stores data
    store_ids = list(range(1, num_stores + 1))
    branches = [random.choice(neighborhood_names) for _ in range(num_stores)]
    
    # Generate coordinates based on each store's branch location
    longitudes = []
    latitudes = []
    for branch in branches:
        lon_min, lon_max, lat_min, lat_max = neighborhoods[branch]
        longitudes.append(round(np.random.uniform(lon_min, lon_max), 4))
        latitudes.append(round(np.random.uniform(lat_min, lat_max), 4))

    stores_data = {
        'store_id': store_ids,
        'store_name': [random.choice(store_names) for _ in range(num_stores)],
        'branch': branches,
        'average_bags_at_9AM': [random.randint(5, 40) for _ in range(num_stores)],
        'average_overall_rating': [round(random.uniform(2.5, 5.0), 1) for _ in range(num_stores)],
        'price': [round(random.uniform(25.0, 100.0), 2) for _ in range(num_stores)],
        'longitude': longitudes,
        'latitude': latitudes,
    }

    stores = pd.DataFrame(stores_data)

    # Overall coordinate range for customers (entire Greater Cairo)
    LON_MIN, LON_MAX = 30.88, 31.52
    LAT_MIN, LAT_MAX = 29.92, 30.11

    # Generate customer data
    customers_data = {
        'customer_id': list(range(1, num_customers + 1)),
        # Random locations within the Greater Cairo area
        'longitude': [round(np.random.uniform(LON_MIN, LON_MAX), 4) for _ in range(num_customers)],
        'latitude': [round(np.random.uniform(LAT_MIN, LAT_MAX), 4) for _ in range(num_customers)]
    }
    
    # Add valuation columns for each store (all at once to avoid fragmentation)
    # Valuation represents how much a customer prefers that store (1-5)
    valuation_data = {
        f'store{sid}_valuation': [random.randint(1, 5) for _ in range(num_customers)]
        for sid in store_ids
    }
    customers_data.update(valuation_data)
    
    customers = pd.DataFrame(customers_data)

    # Save to CSV
    stores.to_csv(os.path.join(output_dir, 'stores.csv'), index=False)
    customers.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)

    return stores, customers
