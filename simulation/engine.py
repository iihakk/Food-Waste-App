"""
Simulation Engine for Food Waste Reduction Platform

This module contains the core simulation logic that models:
- Daily operations of stores listing surprise bags
- Customer behavior and purchasing decisions
- Tracking of KPIs (sold, canceled, wasted bags, revenue, etc.)

The simulation follows this daily cycle:
1. Morning: Each store sets available bags (with random variation from average)
2. Throughout day: Customers arrive and see n stores based on ranking algorithm
3. Customer decision: Pick best store from displayed options (based on valuation)
4. End of day: Calculate sold, canceled, wasted bags and revenue

Key assumptions:
- Each customer buys exactly 1 surprise bag
- Customers only see n stores (not all) - this is the key constraint
- If a customer's preferred stores have no bags, they leave the system
- Stores estimate bags at 9AM but actual availability varies (±30%)

Surprise Bag Model:
- Stores have X items worth of food to distribute
- If N customers order, each bag contains X/N items worth of food
- Example: 10 items, 3 customers -> each bag has ~3.3 items worth
- If 0 customers, all food is wasted
- Revenue = N * bag_price (customers pay per bag, not per item)
"""

import numpy as np
import random
from collections import defaultdict
from datetime import date, timedelta

from simulation.accuracy_tracker import AccuracyTracker
from simulation.cancellation_handler import CancellationHandler


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

    def run(self, num_days, n_stores, ranking_func, shopping_probability=0.7,
            use_accuracy_adjustment=True, alternative_acceptance_rate=0.6):
        """
        Run the simulation for a specified number of days.

        In the surprise bag model:
        - Each store has X "bag equivalents" of food items
        - Each customer who orders gets 1 bag
        - If N customers order, they share all X items (each bag has X/N items)
        - If 0 customers order, all items are wasted

        Args:
            num_days (int): Number of days to simulate
            n_stores (int): Number of stores shown to each customer
            ranking_func (callable): Algorithm that selects which stores to display
                                     Signature: func(stores_df, n, current_bags) -> list[store_id]
            shopping_probability (float): Probability a customer shops on any day (0.0-1.0)
            use_accuracy_adjustment (bool): NEW - Whether to adjust estimates based on history
            alternative_acceptance_rate (float): NEW - Probability customer accepts alternative (0.0-1.0)

        Returns:
            dict: Results containing:
                - store_stats: Per-store metrics
                - daily_data: Day-by-day breakdown
                - store_exposures: How many times each store was shown
                - total_customers: Total customer visits
                - customers_left: Customers who left without buying
                - summary: Aggregated KPIs
                - accuracy_data: NEW - Store accuracy metrics
                - cancellation_data: NEW - Cancellation handling stats
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        store_ids = self.stores['store_id'].tolist()

        # Initialize tracking dictionaries (UPDATED with new fields)
        stats = {sid: {
            'bags_sold': 0,
            'items_available': 0,
            'items_distributed': 0,
            'items_wasted': 0,
            'revenue': 0,
            'potential_revenue': 0,
            'avg_items_per_bag': 0,
            'cancellations': 0,
            'estimated_total': 0,
            'actual_total': 0,
        } for sid in store_ids}

        store_exposures = defaultdict(int)
        total_customers = 0
        customers_left = 0
        daily_data = []

        # Reset handlers for fresh simulation
        self.accuracy_tracker = AccuracyTracker(window_size=14, min_samples=2)
        self.cancellation_handler = CancellationHandler(points_per_cancellation=50)

        for day in range(1, num_days + 1):
            # ==================== MORNING PHASE ====================
            # Each store sets their available items for the day
            daily_estimated = {}  # RENAMED for clarity
            daily_actual = {}     # NEW: Track actual vs estimated
            daily_prices = {}
            
            for _, store in self.stores.iterrows():
                sid = store['store_id']
                estimated = store['average_bags_at_9AM']
                # Add random variation (±30%) - this is the ACTUAL amount
                actual = max(1, int(estimated * np.random.uniform(0.7, 1.3)))
                
                daily_estimated[sid] = estimated
                daily_actual[sid] = actual
                daily_prices[sid] = store['price']
                
                # Track totals for accuracy calculation
                stats[sid]['estimated_total'] += estimated
                stats[sid]['actual_total'] += actual

            # Calculate what the algorithm sees (adjusted or raw)
            if use_accuracy_adjustment and day > 2:
                # Use accuracy-adjusted estimates
                current_items = {}
                for sid in store_ids:
                    adjusted = self.accuracy_tracker.get_adjusted_estimate(sid, daily_estimated[sid])
                    # Conservative: use minimum of adjusted estimate and actual
                    current_items[sid] = min(adjusted, daily_actual[sid])
            else:
                # Use actual items (original behavior)
                current_items = daily_actual.copy()

            # Track what's really available (for cancellation detection)
            actual_remaining = daily_actual.copy()
            
            # Track customers per store
            customers_per_store = defaultdict(int)
            day_customers_left = 0
            day_cancellations = defaultdict(int)

            # Simulation date for priority queue
            sim_date = date(2024, 1, 1) + timedelta(days=day - 1)

            # ==================== NEW: PRIORITY CUSTOMER PHASE ====================
            priority_customers = self.cancellation_handler.get_priority_customers(sim_date)
            
            for priority_info in priority_customers:
                cust_id = priority_info['customer_id']
                priority_store = priority_info['store_id']
                
                if actual_remaining.get(priority_store, 0) > 0:
                    customers_per_store[priority_store] += 1
                    actual_remaining[priority_store] -= 1
                    current_items[priority_store] = max(0, current_items[priority_store] - 1)
                    self.cancellation_handler.serve_priority_customer(sim_date, cust_id, priority_store)
                    total_customers += 1

            # ==================== CUSTOMER ARRIVAL PHASE ====================
            for _, cust in self.customers.iterrows():
                if random.random() > shopping_probability:
                    continue

                total_customers += 1
                cust_id = cust['customer_id']

                # Get stores to display using ranking algorithm
                displayed = ranking_func(self.stores, n_stores, current_items)

                # Track exposures
                for sid in displayed:
                    store_exposures[sid] += 1

                if not displayed:
                    customers_left += 1
                    day_customers_left += 1
                    continue

                # Build customer valuations dict for cancellation handler
                customer_valuations = {}
                for sid in store_ids:
                    col = f'store{sid}_valuation'
                    if col in self.customers.columns:
                        customer_valuations[sid] = cust[col]

                # Customer picks best store from displayed options
                best_store = None
                best_val = 0
                for sid in displayed:
                    val = customer_valuations.get(sid, 0)
                    if val > best_val:
                        best_val = val
                        best_store = sid

                # Customer orders from preferred store
                if best_store:
                    # Check ACTUAL availability (may differ from displayed)
                    if actual_remaining.get(best_store, 0) > 0:
                        # Successful order
                        customers_per_store[best_store] += 1
                        actual_remaining[best_store] -= 1
                        current_items[best_store] = max(0, current_items[best_store] - 1)
                    else:
                        # ============ NEW: CANCELLATION HANDLING ============
                        day_cancellations[best_store] += 1
                        stats[best_store]['cancellations'] += 1
                        
                        # Find stores with actual availability for alternatives
                        available_stores = [sid for sid, bags in actual_remaining.items() if bags > 0]
                        
                        # Handle the cancellation
                        result = self.cancellation_handler.handle_cancellation(
                            customer_id=cust_id,
                            original_store_id=best_store,
                            customer_valuations=customer_valuations,
                            available_stores=available_stores,
                            current_bags=actual_remaining,
                            stores_df=self.stores,
                            order_price=daily_prices[best_store],
                            current_date=sim_date
                        )
                        
                        # Simulate customer decision on alternatives
                        if result['status'] == 'alternatives_available' and result['alternatives']:
                            if random.random() < alternative_acceptance_rate:
                                # Customer accepts alternative
                                alt_store = result['alternatives'][0][0]  # Best alternative
                                
                                if actual_remaining.get(alt_store, 0) > 0:
                                    self.cancellation_handler.accept_alternative(cust_id, alt_store)
                                    customers_per_store[alt_store] += 1
                                    actual_remaining[alt_store] -= 1
                                    current_items[alt_store] = max(0, current_items[alt_store] - 1)
                                else:
                                    # Alternative also ran out
                                    self.cancellation_handler.refuse_alternatives(
                                        cust_id, best_store, daily_prices[best_store], sim_date
                                    )
                                    customers_left += 1
                                    day_customers_left += 1
                            else:
                                # Customer refuses alternatives -> priority + refund + points
                                self.cancellation_handler.refuse_alternatives(
                                    cust_id, best_store, daily_prices[best_store], sim_date
                                )
                                customers_left += 1
                                day_customers_left += 1
                        else:
                            # No alternatives available
                            customers_left += 1
                            day_customers_left += 1
                else:
                    customers_left += 1
                    day_customers_left += 1

            # ==================== END OF DAY PHASE ====================
            # Record accuracy data for each store
            for sid in store_ids:
                self.accuracy_tracker.record_day(
                    sid, 
                    daily_estimated[sid], 
                    daily_actual[sid], 
                    day
                )

            # Calculate results using surprise bag model
            day_stats = {}
            for sid in store_ids:
                items_available = daily_actual[sid]  # Use ACTUAL, not estimated
                num_customers = customers_per_store[sid]
                price = daily_prices[sid]

                if num_customers > 0:
                    bags_sold = num_customers
                    items_distributed = items_available
                    items_wasted = 0
                    items_per_bag = items_available / num_customers
                    revenue = num_customers * price
                else:
                    bags_sold = 0
                    items_distributed = 0
                    items_wasted = items_available
                    items_per_bag = 0
                    revenue = 0

                potential_revenue = items_available * price

                # Update cumulative stats
                stats[sid]['bags_sold'] += bags_sold
                stats[sid]['items_available'] += items_available
                stats[sid]['items_distributed'] += items_distributed
                stats[sid]['items_wasted'] += items_wasted
                stats[sid]['revenue'] += revenue
                stats[sid]['potential_revenue'] += potential_revenue

                day_stats[sid] = {
                    'bags_sold': bags_sold,
                    'items_available': items_available,
                    'items_distributed': items_distributed,
                    'items_wasted': items_wasted,
                    'items_per_bag': round(items_per_bag, 2),
                    'revenue': revenue,
                    'price': price,
                    'num_customers': num_customers,
                    # NEW fields
                    'estimated': daily_estimated[sid],
                    'actual': daily_actual[sid],
                    'cancellations': day_cancellations[sid],
                    'accuracy_ratio': round(daily_actual[sid] / daily_estimated[sid], 3) if daily_estimated[sid] > 0 else 1.0
                }

            daily_data.append({
                'day': day,
                'stores': day_stats,
                'customers_left': day_customers_left,
                'total_cancellations': sum(day_cancellations.values())  # NEW
            })

        # Calculate average items per bag for each store
        for sid in store_ids:
            if stats[sid]['bags_sold'] > 0:
                stats[sid]['avg_items_per_bag'] = round(
                    stats[sid]['items_distributed'] / stats[sid]['bags_sold'], 2
                )

        # Compile results with additional data
        results = {
            'store_stats': stats,
            'daily_data': daily_data,
            'store_exposures': dict(store_exposures),
            'total_customers': total_customers,
            'customers_left': customers_left,
            'summary': self._compute_summary(stats, store_exposures, total_customers, customers_left),
            # NEW: Additional result data
            'accuracy_data': self._compile_accuracy_data(),
            'cancellation_data': self.cancellation_handler.get_statistics()
        }
        return results
    def _compute_summary(self, stats, exposures, total_cust, left):
        """
        Compute aggregated KPIs from simulation results.

        Returns:
            dict: Summary metrics including new accuracy and cancellation metrics
        """
        total_bags_sold = sum(s['bags_sold'] for s in stats.values())
        total_items_available = sum(s['items_available'] for s in stats.values())
        total_items_distributed = sum(s['items_distributed'] for s in stats.values())
        total_items_wasted = sum(s['items_wasted'] for s in stats.values())
        total_revenue = sum(s['revenue'] for s in stats.values())
        total_potential = sum(s['potential_revenue'] for s in stats.values())

        # NEW: Cancellation totals
        total_cancellations = sum(s.get('cancellations', 0) for s in stats.values())

        # Average items per bag across all stores
        avg_items_per_bag = (total_items_distributed / total_bags_sold) if total_bags_sold > 0 else 0

        # Waste rate = items wasted / items available
        waste_rate = (total_items_wasted / total_items_available * 100) if total_items_available > 0 else 0

        # Revenue efficiency = actual revenue / potential revenue
        revenue_efficiency = (total_revenue / total_potential * 100) if total_potential > 0 else 0

        # Cancellation rate
        cancellation_rate = (total_cancellations / total_cust * 100) if total_cust > 0 else 0

        # Average store accuracy
        accuracy_ratios = []
        for sid, s in stats.items():
            if s.get('estimated_total', 0) > 0:
                ratio = s.get('actual_total', 0) / s['estimated_total']
                accuracy_ratios.append(ratio)
        avg_accuracy = np.mean(accuracy_ratios) if accuracy_ratios else 1.0

        # Fairness metric
        exp_vals = list(exposures.values()) if exposures else [0]
        fairness_std = np.std(exp_vals) if len(exp_vals) > 1 else 0

        # Customer satisfaction score (composite metric)
        # Higher is better: penalized by leave rate and cancellation rate
        leave_rate = (left / total_cust * 100) if total_cust > 0 else 0
        satisfaction_score = max(0, 100 - leave_rate - (cancellation_rate * 2))

        return {
            'total_bags_sold': total_bags_sold,
            'total_items_available': total_items_available,
            'total_items_distributed': total_items_distributed,
            'total_items_wasted': total_items_wasted,
            'avg_items_per_bag': round(avg_items_per_bag, 2),
            'total_revenue': round(total_revenue, 2),
            'potential_revenue': round(total_potential, 2),
            'revenue_efficiency': round(revenue_efficiency, 1),
            'waste_rate': round(waste_rate, 1),
            'customer_leave_rate': round(leave_rate, 1),
            'fairness_std': round(fairness_std, 2),
            'total_cancellations': total_cancellations,
            'cancellation_rate': round(cancellation_rate, 2),
            'avg_store_accuracy': round(avg_accuracy, 3),
            'customer_satisfaction_score': round(satisfaction_score, 1)
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
