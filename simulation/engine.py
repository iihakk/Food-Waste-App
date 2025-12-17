"""
Simulation Engine for Food Waste Reduction Platform

This module contains the core simulation logic that models:
- Daily operations of stores listing surprise bags
- Customer behavior and purchasing decisions
- Tracking of KPIs (sold, canceled, wasted bags, revenue, etc.)

The simulation follows this daily cycle:
1. Morning: Each store ESTIMATES bags (this is what the app shows)
2. Throughout day: Customers see n stores based on estimates, make RESERVATIONS
3. End of day: ACTUAL bags revealed, cancellations if actual < reservations
4. Revenue/waste calculated based on actual fulfillment

UNIFIED LOST REVENUE DEFINITION:
================================
- Unsold bags = Actual bags that weren't sold (waste)
- Lost Revenue = Unsold bags × Price

Example: Store has 10 actual bags, got 7 reservations
→ fulfilled=7, unsold=3, lost_revenue = 3 × price

Example: Store has 10 actual bags, got 12 reservations
→ fulfilled=10, cancelled=2, unsold=0, lost_revenue = 0
(Cancelled orders are NOT lost revenue - those bags never existed)

Key assumptions:
- Algorithm and customers only see ESTIMATED bags (not actual)
- Customers make reservations, not immediate purchases
- Actual bags revealed at end of day (varies ±30% from estimate)
- If actual < reservations → some orders cancelled (NOT lost revenue)
- If actual >= reservations → all orders fulfilled, excess = unsold = waste

Surprise Bag Model:
- Each fulfilled reservation = 1 bag sold
- Unsold bags (actual - fulfilled) = waste
- Revenue = fulfilled_orders × bag_price
- Lost Revenue = unsold_bags × bag_price
"""

import numpy as np
import random
from collections import defaultdict
from datetime import date, timedelta

from simulation.accuracy_tracker import AccuracyTracker
from simulation.cancellation_handler import CancellationHandler

# Import GA callbacks for end-of-day evaluation
try:
    from simulation.algorithms import ga_start_day, ga_end_day
    _GA_CALLBACKS_AVAILABLE = True
except ImportError:
    _GA_CALLBACKS_AVAILABLE = False


class SimulationEngine:
    """
    Core simulation engine for the food waste reduction marketplace.

    Attributes:
        stores (DataFrame): Store data (id, name, branch, bags, rating, price, location)
        customers (DataFrame): Customer data (id, location, valuations for each store)
        seed (int): Random seed for reproducibility
    """

    def __init__(self, stores_df, customers_df, seed=42):
        """
        Initialize the simulation engine.

        Args:
            stores_df (DataFrame): Store information
            customers_df (DataFrame): Customer information with store valuations
            seed (int): Random seed for reproducible results (default: 42)
        """
        self.stores = stores_df.copy()
        self.customers = customers_df.copy()
        self.seed = seed

    def _haversine_distance(self, coord1, coord2):
        """
        Calculate distance between two lat/lon coordinates in km.

        Args:
            coord1: (latitude, longitude) tuple
            coord2: (latitude, longitude) tuple

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth radius in km

        lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
        lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def run(self, num_days, n_stores, ranking_func, shopping_probability=0.7,
            use_accuracy_adjustment=True, alternative_acceptance_rate=0.6):
        """
        Run the simulation for a specified number of days.

        CORRECT MODEL:
        - Morning: Stores provide ESTIMATES (what algorithm/customers see)
        - Day: Customers make RESERVATIONS based on estimates
        - End of Day: ACTUAL bags revealed
          - If actual >= reservations: all fulfilled
          - If actual < reservations: some cancelled
          - Unsold = actual - fulfilled = waste

        Args:
            num_days (int): Number of days to simulate
            n_stores (int): Number of stores shown to each customer
            ranking_func (callable): Algorithm that selects which stores to display
                                     Signature: func(stores_df, n, current_bags) -> list[store_id]
            shopping_probability (float): Probability a customer shops on any day (0.0-1.0)
            use_accuracy_adjustment (bool): Whether to adjust estimates based on history (future use)
            alternative_acceptance_rate (float): Probability customer accepts alternative when cancelled

        Returns:
            dict: Results containing all KPIs and the Algorithm Score
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        store_ids = self.stores['store_id'].tolist()

        # Initialize tracking
        stats = {sid: {
            'reservations': 0,
            'fulfilled': 0,
            'cancelled': 0,
            'unsold': 0,
            'revenue': 0,
            'lost_revenue': 0,
            'estimated_total': 0,
            'actual_total': 0,
        } for sid in store_ids}

        store_exposures = defaultdict(int)
        total_customers = 0
        customers_left = 0
        daily_data = []

        for day in range(1, num_days + 1):
            # ==================== MORNING PHASE ====================
            # Store sets ESTIMATE (what app shows) and ACTUAL (revealed end of day)
            daily_estimated = {}
            daily_actual = {}
            daily_prices = {}

            for _, store in self.stores.iterrows():
                sid = store['store_id']
                estimated = store['average_bags_at_9AM']
                # Actual varies ±30% from estimate (revealed at end of day)
                actual = max(1, int(estimated * np.random.uniform(0.7, 1.3)))

                daily_estimated[sid] = estimated
                daily_actual[sid] = actual
                daily_prices[sid] = store['price']

                stats[sid]['estimated_total'] += estimated
                stats[sid]['actual_total'] += actual

            # Notify GA of day start (for shadow simulation tracking)
            if _GA_CALLBACKS_AVAILABLE:
                ga_start_day(daily_estimated, self.stores)

            # Algorithm sees ESTIMATED bags (not actual!)
            remaining_estimated = daily_estimated.copy()

            # Track reservations per store for this day
            daily_reservations = defaultdict(int)
            day_customers_left = 0

            # ==================== CUSTOMER ARRIVAL PHASE ====================
            # Customers make RESERVATIONS based on estimated availability
            for _, cust in self.customers.iterrows():
                if random.random() > shopping_probability:
                    continue

                total_customers += 1
                cust_id = cust['customer_id']

                # Extract customer location (for distance-aware algorithms)
                customer_location = None
                if 'latitude' in cust and 'longitude' in cust:
                    customer_location = (cust['latitude'], cust['longitude'])

                # Build customer valuations FIRST (needed for personalized algorithms)
                customer_valuations = {}
                for sid in store_ids:
                    col = f'store{sid}_valuation'
                    if col in self.customers.columns:
                        customer_valuations[sid] = cust[col]

                # Algorithm sees remaining ESTIMATES + customer preferences + location
                # Location-aware algorithms can use customer_location for distance-based ranking
                displayed = ranking_func(self.stores, n_stores, remaining_estimated,
                                        customer_valuations=customer_valuations,
                                        customer_location=customer_location)

                # Track exposures
                for sid in displayed:
                    store_exposures[sid] += 1

                if not displayed:
                    customers_left += 1
                    day_customers_left += 1
                    continue

                # Customer picks best store from displayed options
                # Factor in both valuation AND distance (closer stores preferred)
                best_store = None
                best_score = -1
                for sid in displayed:
                    val = customer_valuations.get(sid, 0)
                    if remaining_estimated.get(sid, 0) > 0:
                        # Calculate distance factor if location available
                        distance_factor = 1.0
                        if customer_location:
                            store_row = self.stores[self.stores['store_id'] == sid].iloc[0]
                            if 'latitude' in store_row and 'longitude' in store_row:
                                dist = self._haversine_distance(
                                    customer_location,
                                    (store_row['latitude'], store_row['longitude'])
                                )
                                # Closer stores get higher factor (decay over distance)
                                # At 0km: factor=1.0, at 5km: factor~0.37, at 10km: factor~0.14
                                distance_factor = np.exp(-dist / 5.0)

                        # Combined score: valuation * distance_factor
                        score = val * (0.7 + 0.3 * distance_factor)
                        if score > best_score:
                            best_score = score
                            best_store = sid

                # Customer makes RESERVATION (not purchase yet)
                if best_store and remaining_estimated.get(best_store, 0) > 0:
                    daily_reservations[best_store] += 1
                    remaining_estimated[best_store] -= 1  # Decrement ESTIMATE
                else:
                    customers_left += 1
                    day_customers_left += 1

            # ==================== END OF DAY PHASE ====================
            # Actual bags revealed, process fulfillment vs cancellations
            day_stats = {}
            day_cancelled = 0

            for sid in store_ids:
                estimated = daily_estimated[sid]
                actual = daily_actual[sid]
                reservations = daily_reservations[sid]
                price = daily_prices[sid]

                # ============================================================
                # UNIFIED LOST REVENUE LOGIC
                # ============================================================
                # Unsold bags = Actual bags that weren't sold (waste)
                # Lost Revenue = Unsold bags × Price
                #
                # Example: Store has 10 actual bags, got 7 reservations
                # - fulfilled = 7, unsold = 3, lost_revenue = 3 × price
                #
                # Example: Store has 10 actual bags, got 12 reservations  
                # - fulfilled = 10, cancelled = 2, unsold = 0, lost_revenue = 0
                # (Cancelled orders are NOT lost revenue - those bags never existed)
                # ============================================================
                
                if reservations <= actual:
                    # All reservations fulfilled, excess bags = waste
                    fulfilled = reservations
                    cancelled = 0
                    unsold = actual - reservations
                else:
                    # Not enough actual bags - some orders cancelled, NO waste
                    fulfilled = actual
                    cancelled = reservations - actual
                    unsold = 0

                revenue = fulfilled * price
                lost_revenue = unsold * price  # UNIFIED: Lost Revenue = Unsold × Price

                # Update cumulative stats
                stats[sid]['reservations'] += reservations
                stats[sid]['fulfilled'] += fulfilled
                stats[sid]['cancelled'] += cancelled
                stats[sid]['unsold'] += unsold
                stats[sid]['revenue'] += revenue
                stats[sid]['lost_revenue'] += lost_revenue

                day_cancelled += cancelled

                day_stats[sid] = {
                    'estimated': estimated,
                    'actual': actual,
                    'reservations': reservations,
                    'fulfilled': fulfilled,
                    'cancelled': cancelled,
                    'unsold': unsold,
                    'revenue': revenue,
                    'price': price
                }

            # NOTE: Cancelled customers are tracked separately via cancellation_rate
            # Don't add to customers_left to avoid double-counting in satisfaction score

            # Notify GA of day end (for fitness evaluation and evolution)
            if _GA_CALLBACKS_AVAILABLE:
                ga_end_day(daily_actual, daily_prices)

            daily_data.append({
                'day': day,
                'stores': day_stats,
                'customers_left': day_customers_left,
                'total_cancellations': day_cancelled
            })

        # Compile results
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
        """
        Compute aggregated KPIs from simulation results.

        NEW RESERVATION MODEL METRICS:
        - reservations: Total orders placed
        - fulfilled: Orders successfully completed
        - cancelled: Orders cancelled due to insufficient actual bags
        - unsold: Actual bags that weren't sold (waste)

        Returns:
            dict: Summary metrics including Algorithm Score
        """
        # Core metrics from new model
        total_reservations = sum(s['reservations'] for s in stats.values())
        total_fulfilled = sum(s['fulfilled'] for s in stats.values())
        total_cancelled = sum(s['cancelled'] for s in stats.values())
        total_unsold = sum(s['unsold'] for s in stats.values())
        total_revenue = sum(s['revenue'] for s in stats.values())
        total_lost_revenue = sum(s['lost_revenue'] for s in stats.values())
        total_estimated = sum(s['estimated_total'] for s in stats.values())
        total_actual = sum(s['actual_total'] for s in stats.values())

        # Revenue efficiency: measure how much of AVAILABLE inventory was sold
        # This measures "of bags available (actual), what % were sold?"
        #
        # total_potential = revenue + lost_revenue (from unsold bags)
        #                 = fulfilled * price + unsold * price
        #                 = actual * price (total value of available inventory)
        #
        # If 0 waste → efficiency = 100% (all bags sold)
        # If some waste → efficiency < 100%
        total_potential = total_revenue + total_lost_revenue

        # Fulfillment rate = fulfilled / reservations
        fulfillment_rate = (total_fulfilled / total_reservations * 100) if total_reservations > 0 else 0

        # Cancellation rate = cancelled / reservations
        cancellation_rate = (total_cancelled / total_reservations * 100) if total_reservations > 0 else 0

        # Waste rate = unsold / total actual bags
        waste_rate = (total_unsold / total_actual * 100) if total_actual > 0 else 0

        # Revenue efficiency = actual revenue / potential revenue
        # Potential = value of all available bags (actual inventory)
        # If no waste → efficiency = 100% (all bags sold)
        # If some waste → efficiency < 100%
        revenue_efficiency = (total_revenue / total_potential * 100) if total_potential > 0 else 100

        # Customer leave rate (including those who left + cancelled orders)
        leave_rate = (left / total_cust * 100) if total_cust > 0 else 0

        # Fairness metric - coefficient of variation (std/mean) of exposures
        # Lower CV = more evenly distributed exposures = fairer
        exp_vals = list(exposures.values()) if exposures else [0]
        fairness_std = np.std(exp_vals) if len(exp_vals) > 1 else 0
        mean_exp = np.mean(exp_vals) if exp_vals else 1
        coef_variation = (fairness_std / mean_exp) if mean_exp > 0 else 0
        fairness_normalized = coef_variation * 100  # Convert to percentage

        # =====================================================
        # ALGORITHM SCORE - Composite metric for comparison
        # =====================================================
        # Formula: Weighted combination of key objectives
        #
        # Components (all normalized to 0-100 scale):
        # 1. Revenue Performance (30%): How much potential revenue captured
        # 2. Waste Reduction (30%): Inverse of waste rate
        # 3. Customer Satisfaction (25%): Based on fulfillment and leave rates
        # 4. Fairness (15%): How evenly stores are exposed
        #
        # Higher score = better algorithm performance

        # Component 1: Revenue Performance (0-100)
        revenue_score = revenue_efficiency  # Already 0-100

        # Component 2: Demand Fulfillment (0-100)
        # Since supply > demand, absolute waste is structurally constrained
        # Measure how many customers were served instead
        demand_fulfillment = (total_fulfilled / total_cust * 100) if total_cust > 0 else 0
        waste_score = demand_fulfillment  # Higher fulfillment = higher score

        # Component 3: Customer Satisfaction (0-100)
        # Penalize for cancellations and customers leaving
        satisfaction_score = max(0, 100 - leave_rate - (cancellation_rate * 1.5))

        # Component 4: Fairness (0-100)
        # Lower std deviation = higher fairness score
        fairness_score = max(0, 100 - fairness_normalized)

        # Weighted Algorithm Score
        algorithm_score = (
            0.30 * revenue_score +
            0.30 * waste_score +
            0.25 * satisfaction_score +
            0.15 * fairness_score
        )

        return {
            # Core counts
            'total_reservations': total_reservations,
            'total_fulfilled': total_fulfilled,
            'total_cancelled': total_cancelled,
            'total_unsold': total_unsold,

            # Rates
            'fulfillment_rate': round(fulfillment_rate, 1),
            'cancellation_rate': round(cancellation_rate, 1),
            'waste_rate': round(waste_rate, 1),
            'customer_leave_rate': round(leave_rate, 1),
            'demand_fulfillment': round(demand_fulfillment, 1),

            # Revenue
            'total_revenue': round(total_revenue, 2),
            'total_lost_revenue': round(total_lost_revenue, 2),
            'potential_revenue': round(total_potential, 2),
            'revenue_efficiency': round(revenue_efficiency, 1),

            # Fairness
            'fairness_std': round(fairness_std, 2),
            'fairness_score': round(fairness_score, 1),

            # Composite scores
            'satisfaction_score': round(satisfaction_score, 1),
            'algorithm_score': round(algorithm_score, 1),

            # Legacy compatibility (for app.py charts)
            'total_bags_sold': total_fulfilled,
            'total_items_wasted': total_unsold,
            'total_items_distributed': total_fulfilled,
            'avg_items_per_bag': 1.0,  # Each reservation = 1 bag in this model
        }

    def _compile_accuracy_data(self):
        """
        Compile accuracy tracking data for all stores.

        Returns:
            dict: {store_id: accuracy_metrics}
        """
        accuracy_data = {}
        for store_id in self.accuracy_tracker.history.keys():
            accuracy_data[store_id] = self.accuracy_tracker.get_store_performance_summary(store_id)
        return accuracy_data
