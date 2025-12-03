import numpy as np
import random
from collections import defaultdict

class SimulationEngine:
    def __init__(self, stores_df, customers_df, seed=42):
        self.stores = stores_df.copy()
        self.customers = customers_df.copy()
        self.seed = seed

    def run(self, num_days, n_stores, ranking_func, shopping_probability=0.7):
        random.seed(self.seed)
        np.random.seed(self.seed)

        store_ids = self.stores['store_id'].tolist()
        valuation_cols = [c for c in self.customers.columns if '_valuation' in c]

        # tracking
        stats = {sid: {'sold': 0, 'canceled': 0, 'wasted': 0, 'revenue': 0, 'lost_revenue': 0}
                 for sid in store_ids}
        store_exposures = defaultdict(int)
        total_customers = 0
        customers_left = 0
        daily_data = []

        for day in range(1, num_days + 1):
            # morning: setup daily bags and prices
            daily_bags = {}
            daily_prices = {}
            for _, store in self.stores.iterrows():
                sid = store['store_id']
                est = store['average_bags_at_9AM']
                daily_bags[sid] = max(1, int(est * np.random.uniform(0.7, 1.3)))
                daily_prices[sid] = round(random.uniform(25.0, 75.0), 2)

            current_bags = daily_bags.copy()
            demand = defaultdict(int)
            day_customers_left = 0

            # customer arrivals
            for _, cust in self.customers.iterrows():
                if random.random() > shopping_probability:
                    continue

                total_customers += 1

                # get displayed stores from ranking algorithm
                displayed = ranking_func(self.stores, n_stores, current_bags)
                for sid in displayed:
                    store_exposures[sid] += 1

                if not displayed:
                    customers_left += 1
                    day_customers_left += 1
                    continue

                # customer picks from displayed stores based on valuation
                best_store = None
                best_val = 0
                for sid in displayed:
                    col = f'store{sid}_valuation'
                    if col in self.customers.columns:
                        val = cust[col]
                        if val > best_val:
                            best_val = val
                            best_store = sid

                if best_store and current_bags.get(best_store, 0) > 0:
                    demand[best_store] += 1
                    current_bags[best_store] -= 1
                else:
                    customers_left += 1
                    day_customers_left += 1

            # end of day calculations
            day_stats = {}
            for sid in store_ids:
                actual = daily_bags[sid]
                d = demand[sid]
                price = daily_prices[sid]

                sold = min(actual, d)
                canceled = max(0, d - actual)
                wasted = max(0, actual - d)
                rev = sold * price
                lost = wasted * price

                stats[sid]['sold'] += sold
                stats[sid]['canceled'] += canceled
                stats[sid]['wasted'] += wasted
                stats[sid]['revenue'] += rev
                stats[sid]['lost_revenue'] += lost

                day_stats[sid] = {
                    'sold': sold, 'canceled': canceled, 'wasted': wasted,
                    'revenue': rev, 'lost_revenue': lost, 'price': price,
                    'actual_bags': actual, 'demand': d
                }

            daily_data.append({
                'day': day,
                'stores': day_stats,
                'customers_left': day_customers_left
            })

        # compile results
        results = {
            'store_stats': stats,
            'daily_data': daily_data,
            'store_exposures': dict(store_exposures),
            'total_customers': total_customers,
            'customers_left': customers_left,
            'summary': self._compute_summary(stats, store_exposures, total_customers, customers_left)
        }
        return results

    def _compute_summary(self, stats, exposures, total_cust, left):
        total_sold = sum(s['sold'] for s in stats.values())
        total_canceled = sum(s['canceled'] for s in stats.values())
        total_wasted = sum(s['wasted'] for s in stats.values())
        total_revenue = sum(s['revenue'] for s in stats.values())
        total_lost = sum(s['lost_revenue'] for s in stats.values())

        demand = total_sold + total_canceled
        available = total_sold + total_wasted

        # fairness: std dev of exposures (lower = fairer)
        exp_vals = list(exposures.values()) if exposures else [0]
        fairness_std = np.std(exp_vals) if len(exp_vals) > 1 else 0

        return {
            'total_sold': total_sold,
            'total_canceled': total_canceled,
            'total_wasted': total_wasted,
            'total_revenue': round(total_revenue, 2),
            'total_lost_revenue': round(total_lost, 2),
            'fulfillment_rate': round(total_sold / demand * 100, 1) if demand > 0 else 0,
            'waste_rate': round(total_wasted / available * 100, 1) if available > 0 else 0,
            'customer_leave_rate': round(left / total_cust * 100, 1) if total_cust > 0 else 0,
            'fairness_std': round(fairness_std, 2)
        }
