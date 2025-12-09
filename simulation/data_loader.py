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

#these params are the deafult if the box isn't checked

def generate_sample_data(output_dir='data', seed=44, num_stores=10, num_customers=150):
    """
    Generate sample store and customer data for testing.



    This function creates realistic sample data with:
    - Configurable number of stores across Cairo neighborhoods
    - Configurable number of customers with random locations and valuations

    Note: This is for initial testing only. For the actual simulation,
    use the manually curated stores.csv with Egyptian bakeries.

    Args:
        output_dir (str): Directory to save CSV files
        seed (int): Random seed for reproducibility
        num_stores (int): Number of stores to generate
        num_customers (int): Number of customers to generate

    Returns:
        tuple: (stores_df, customers_df) DataFrames
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Extensive store name pool (Egyptian and international brands)
    store_names = [
        # Egyptian bakeries and cafes
        'La Poire', 'El Abd', 'Mandarine Koueider', 'Etoile', 'Tseppas',
        'El Malky', 'Nola Cupcakes', 'Dukes', 'Left Bank', 'Cake House',
        'Sweet Tooth', 'Cairo Kitchen', 'Zooba', 'Kazouza', 'Abou El Sid',
        'El Fishawy', 'Groppi', 'Simonds', 'Delices', 'Andrea',
        # International chains
        'TBS', 'Dunkin', 'Costa Coffee', 'Starbucks', 'Paul',
        'Cilantro', "Beano's", 'Harris Cafe', 'Cinnabon', 'Krispy Kreme',
        'Caribou Coffee', 'Tim Hortons', 'McCafe', 'Gloria Jeans',
        'The Coffee Bean', 'Peet\'s Coffee', 'Second Cup', 'Aroma Espresso',
        # More local options
        'Bread Basket', 'French Bakery', 'Hot Bread Kitchen', 'Chez Michel',
        'Le Pacha', 'Bakery House', 'Croissant Show', 'Patisserie Royale',
        'Golden Wheat', 'Cairo Bakes', 'Flour Power', 'Sugar & Spice',
        'The Bakery', 'Fresh Bites', 'Sweet Dreams', 'Pastry Paradise'
    ]
    
    # Cairo neighborhoods/areas with coordinate ranges (lon_min, lon_max, lat_min, lat_max)
    branch_coord_ranges = {
        'Zamalek': (31.21, 31.23, 30.05, 30.07),
        'New Cairo': (31.45, 31.55, 29.99, 30.05),
        'Maadi': (31.24, 31.28, 29.94, 29.98),
        'Heliopolis': (31.32, 31.36, 30.08, 30.11),
        'Mohandessin': (31.19, 31.22, 30.04, 30.07),
        'Downtown': (31.23, 31.26, 30.04, 30.06),
        'Nasr City': (31.32, 31.38, 30.05, 30.09),
        '6th October': (30.90, 31.00, 29.93, 30.00),
        'Dokki': (31.20, 31.22, 30.03, 30.06),
        'Giza': (31.19, 31.22, 29.99, 30.03),
        'Tagamo3': (31.40, 31.48, 30.00, 30.04),
        'Sheikh Zayed': (30.94, 31.02, 30.01, 30.06),
        'Rehab City': (31.48, 31.52, 30.05, 30.08),
        'Obour City': (31.45, 31.50, 30.14, 30.18),
        'Shorouk City': (31.58, 31.63, 30.10, 30.14),
        'Hadayek El Kobba': (31.28, 31.32, 30.08, 30.11),
        'Ain Shams': (31.30, 31.34, 30.10, 30.13),
        'Shubra': (31.23, 31.27, 30.07, 30.11),
        'Imbaba': (31.20, 31.23, 30.07, 30.10),
        'Agouza': (31.20, 31.22, 30.05, 30.07),
        'Garden City': (31.22, 31.24, 30.02, 30.04),
        'Korba': (31.32, 31.34, 30.08, 30.10),
        'Roxy': (31.30, 31.33, 30.08, 30.10),
        'Almaza': (31.34, 31.36, 30.09, 30.11),
        'El Marg': (31.34, 31.38, 30.12, 30.16),
        'Madinet Nasr': (31.32, 31.38, 30.04, 30.08),
        'El Haram': (31.12, 31.16, 29.98, 30.02),
        'Faisal': (31.15, 31.19, 29.99, 30.03),
        'El Manial': (31.22, 31.24, 30.00, 30.02),
        'Rod El Farag': (31.23, 31.26, 30.08, 30.11)
    }
    
    branches = list(branch_coord_ranges.keys())

    # Generate stores data
    store_ids = list(range(1, num_stores + 1))
    selected_names = [random.choice(store_names) for _ in range(num_stores)]
    selected_branches = [random.choice(branches) for _ in range(num_stores)]
    
    # Generate coordinates within branch ranges
    longitudes = []
    latitudes = []
    for branch in selected_branches:
        lon_min, lon_max, lat_min, lat_max = branch_coord_ranges[branch]
        longitudes.append(round(np.random.uniform(lon_min, lon_max), 4))
        latitudes.append(round(np.random.uniform(lat_min, lat_max), 4))
    
    stores = pd.DataFrame({
        'store_id': store_ids,
        'store_name': selected_names,
        'branch': selected_branches,
        'average_bags_at_9AM': [random.randint(5, 30) for _ in range(num_stores)],
        'average_overall_rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(num_stores)],
        'price': [round(random.uniform(25.0, 75.0), 2) for _ in range(num_stores)],
        'longitude': longitudes,
        'latitude': latitudes
    })

    # Generate customer data efficiently using numpy for large datasets
    customers = pd.DataFrame({
        'customer_id': list(range(1, num_customers + 1)),
        # Random locations within greater Cairo area
        'longitude': np.round(np.random.uniform(31.0, 31.6, num_customers), 4),
        'latitude': np.round(np.random.uniform(29.9, 30.2, num_customers), 4)
    })

    # Add valuation column for each store
    # Using numpy for efficiency with large datasets
    for sid in stores['store_id']:
        customers[f'store{sid}_valuation'] = np.random.randint(1, 6, num_customers)

    # Save to CSV
    stores.to_csv(os.path.join(output_dir, 'stores.csv'), index=False)
    customers.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)

    return stores, customers
