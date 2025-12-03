import pandas as pd
import numpy as np
import random
import os

def load_data(stores_path, customers_path):
    stores = pd.read_csv(stores_path)
    customers = pd.read_csv(customers_path)
    return stores, customers

def generate_sample_data(output_dir='data', seed=44):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # stores data
    stores = pd.DataFrame({
        'store_id': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'store_name': ['TBS', 'Dunkin', 'Costa Coffee', 'Starbucks', 'Paul',
                       'TBS', 'Cilantro', "Beano's", 'Harris Cafe', 'Dunkin'],
        'branch': ['Zamalek', 'New Cairo', 'Maadi', 'Heliopolis', 'Mohandessin',
                   'New Cairo', 'Zamalek', 'Maadi', 'Heliopolis', 'Mohandessin'],
        'average_bags_at_9AM': [random.randint(5, 30) for _ in range(10)],
        'average_overall_rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(10)],
        'price': [round(random.uniform(25.0, 75.0), 2) for _ in range(10)],
        'longitude': [31.2194, 31.4913, 31.2625, 31.3375, 31.2001,
                      31.4701, 31.2243, 31.2587, 31.3421, 31.2056],
        'latitude': [30.0626, 30.0171, 29.9602, 30.0911, 30.0561,
                     30.0285, 30.0651, 29.9711, 30.0875, 30.0512]
    })

    # customers data
    num_customers = 150
    customers = pd.DataFrame({
        'customer_id': list(range(1, num_customers + 1)),
        'longitude': [round(31.2 + np.random.uniform(-0.1, 0.3), 4) for _ in range(num_customers)],
        'latitude': [round(30.0 + np.random.uniform(-0.1, 0.1), 4) for _ in range(num_customers)]
    })

    # add valuation for each store
    for sid in stores['store_id']:
        customers[f'store{sid}_valuation'] = [random.randint(1, 5) for _ in range(num_customers)]

    stores.to_csv(os.path.join(output_dir, 'stores.csv'), index=False)
    customers.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)

    return stores, customers
