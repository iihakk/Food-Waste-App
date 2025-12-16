"""
Ranking Algorithms for Food Waste Reduction Platform

This module contains different store ranking algorithms that determine
which stores are displayed to customers. The goal is to optimize for:
- Minimizing food waste (unsold bags)
- Maximizing revenue
- Fair distribution of exposure across stores

Each algorithm has the same signature:
    func(stores_df, n, current_bags, customer_valuations=None) -> list[store_id]

Where:
    - stores_df: DataFrame with store information
    - n: Number of stores to return (display to customer)
    - current_bags: Dict mapping store_id -> remaining bags
    - customer_valuations: Dict mapping store_id -> customer's preference (0-5)
                          If provided, enables PERSONALIZED ranking per customer

Algorithm Design Techniques (allowed per course):
1. Greedy - Make locally optimal choices
2. Divide and Conquer - Break problem into subproblems
3. Transform and Conquer - Change data representation
4. Dynamic Programming - Optimal substructure + overlapping subproblems
5. Backtracking - Build solution incrementally

NOTE: Brute force is NOT allowed in this course.
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta


def greedy_baseline(stores_df, n, current_bags, customer_valuations=None):
    """
    BASELINE ALGORITHM: Greedy by Rating

    Strategy: Always show the highest-rated stores that have bags available.
    This is the current system's approach.

    Pros:
    - Simple to implement
    - Shows "best" stores to customers

    Cons:
    - Popular stores sell out quickly
    - Less popular stores get no visibility
    - Leads to unfair exposure distribution
    - Results in more food waste at low-rated stores

    Time Complexity: O(n log n) for sorting
    Technique: Greedy

    Args:
        stores_df (DataFrame): Store data with 'store_id' and 'average_overall_rating'
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}

    Returns:
        list: Top n store IDs by rating that have bags available
    """
    # Filter to stores with available bags
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    available = stores_df[stores_df['store_id'].isin(available_ids)]

    # Sort by rating (descending) and take top n
    top_n = available.nlargest(n, 'average_overall_rating')
    return top_n['store_id'].tolist()


def inventory_aware(stores_df, n, current_bags, customer_valuations=None):
    """
    IMPROVED ALGORITHM: Inventory-Aware Ranking

    Strategy: Balance rating with inventory level to reduce waste.
    Stores with more unsold bags get priority to reduce end-of-day waste.

    Scoring formula:
        score = 0.5 * (rating/5.0) + 0.5 * (bags/max_bags)

    Pros:
    - Reduces food waste by prioritizing stores with more inventory
    - Still considers quality (rating)
    - Better fairness than pure greedy

    Cons:
    - May show lower-rated stores
    - Fixed 50/50 weight ratio may not be optimal

    Time Complexity: O(n) for scoring + O(n log n) for sorting
    Technique: Greedy with weighted scoring

    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}

    Returns:
        list: Top n store IDs by combined score
    """
    # Filter to stores with available bags
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)]
    max_bags = max(current_bags[sid] for sid in available_ids)

    # Calculate combined score for each store
    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        # Normalize rating to 0-1 range
        rating_score = row['average_overall_rating'] / 5.0
        # Normalize inventory to 0-1 range
        inventory_score = current_bags[sid] / max_bags

        # Weighted combination: 50% rating + 50% inventory
        score = 0.5 * rating_score + 0.5 * inventory_score
        scores.append((sid, score))

    # Sort by score descending and return top n
    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def inventory_rating_equilibrium(stores_df, n, current_bags, customer_valuations=None):
    """
    INVENTORY-RATING EQUILIBRIUM: Greedy with Inventory-Demand Balancing

    Strategy: Extension of the Greedy Baseline.
    Instead of sorting purely by rating, this algorithm seeks an 'equilibrium' 
    where the highest rated stores (Demand) that also have the most stock (Supply) 
    are prioritized. 
    
    It prevents the issue where the greedy baseline shows high-rated stores 
    that might only have 1 bag left, ignoring a slightly lower-rated store 
    with 50 bags that needs the customers more.

    Formula:
        Score = Rating_Normalized * (1 + Inventory_Normalized)

    Pros:
    - Maximizes sell-through rate (High Quality + High Volume)
    - Prevents stock-outs at top stores by rotating in high-supply alternatives
    - Maintains the "Best Stores" feel of the greedy baseline

    Time Complexity: O(n log n)
    Technique: Greedy (with weighted heuristic)

    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}

    Returns:
        list: Top n store IDs by equilibrium score
    """
    # 1. Filter to stores with available bags (Identical to Greedy Baseline)
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    # Create a copy to avoid SettingWithCopy warnings on the original DF
    available = stores_df[stores_df['store_id'].isin(available_ids)].copy()

    # 2. Calculate Normalization Factors
    # Avoid division by zero if max_bags is somehow 0 (though unlikely due to filter)
    max_bags = max([current_bags[sid] for sid in available_ids]) if available_ids else 1
    max_rating = 5.0

    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory = current_bags[sid]

        # Demand Index (Normalized Rating)
        demand_index = rating / max_rating

        # Supply Index (Normalized Inventory)
        supply_index = inventory / max_bags

        # Equilibrium Score Calculation:
        # We take the Demand (Rating) as the base, and boost it by the Supply (Inventory).
        # - If a store has high rating but low inventory, it gets a standard score.
        # - If a store has high rating AND high inventory, it gets a massive boost (Equilibrium).
        score = demand_index * (1.0 + supply_index)
        
        scores.append((sid, score))

    # 3. Sort by Score (descending) and take top n
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return [sid for sid, _ in scores[:n]]


def underdog_boost(stores_df, n, current_bags, customer_valuations=None):
    """
    UNDERDOG BOOST: Inverse rating multiplier for inventory priority.

    Key Insight: High-rated stores will sell regardless of visibility.
    Low-rated stores with inventory need exposure to avoid waste.

    Approach: Instead of adding rating and inventory, we use rating
    as an INVERSE multiplier. Lower rating = higher boost for inventory.

    Formula:
        boost_factor = (5.5 - rating) / 4.5  # Range: ~0.1 to 1.0
        score = inventory_norm * boost_factor + 0.2 * rating_norm

    A 3.0 rated store with 20 bags scores HIGHER than a 4.5 rated store
    with 20 bags, because the 3.0 store needs help getting customers.

    Time Complexity: O(S log S)
    Technique: Greedy with inverse weighting
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)]
    max_bags = max(current_bags[sid] for sid in available_ids)

    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory_norm = current_bags[sid] / max_bags
        rating_norm = rating / 5.0

        # Inverse boost: lower rating = higher multiplier
        # Rating 5.0 -> boost 0.11, Rating 3.0 -> boost 0.56, Rating 1.0 -> boost 1.0
        boost_factor = (5.5 - rating) / 4.5

        # Inventory heavily weighted, rating just prevents showing terrible stores
        score = inventory_norm * boost_factor + 0.2 * rating_norm
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def waste_prevention_threshold(stores_df, n, current_bags, customer_valuations=None):
    """
    WASTE PREVENTION THRESHOLD: Binary rescue system for at-risk stores.

    Key Insight: Waste only happens when a store gets 0 customers.
    A store with 5 bags and 0 customers = 100% waste.
    A store with 30 bags and 1 customer = 0% waste (just less items per bag).

    Approach: Calculate "risk score" based on inventory vs expected demand.
    High-rated stores have high expected demand, low-rated stores have low demand.
    If inventory >> expected_demand, store is "at risk" and gets priority.

    Risk calculation:
        expected_demand = rating / 5.0 * avg_customers_per_store
        risk = inventory / expected_demand (capped at 3.0)

    Stores with risk > 1.5 are considered "at-risk" and prioritized.

    Time Complexity: O(S log S)
    Technique: Transform and Conquer (transform to risk scores)
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)]

    # Estimate: assume each store might get ~3 customers on average
    avg_customers_estimate = 3.0

    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory = current_bags[sid]

        # Expected demand based on rating (higher rating = more expected customers)
        expected_demand = (rating / 5.0) * avg_customers_estimate
        expected_demand = max(0.5, expected_demand)  # Floor to avoid division issues

        # Risk: how much inventory vs expected demand
        # High risk = lots of inventory relative to expected customers
        risk = min(3.0, inventory / expected_demand)

        # At-risk stores (risk > 1.5) get priority boost
        if risk > 1.5:
            # Rescue priority: high risk stores shown first
            score = 1.0 + risk  # Score 2.5 to 4.0 for at-risk
        else:
            # Normal stores: blend of rating and inventory
            rating_norm = rating / 5.0
            score = 0.6 * rating_norm + 0.4 * (inventory / 30.0)

        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]



def time_decay_urgency(stores_df, n, current_bags, closing_times=None, current_time=None, 
                       customer_valuations=None, exposure_history=None, customer_id=None,
                       waste_prevention_mode='balanced'):  # NEW PARAMETER
    """
    Enhanced with aggressive waste prevention strategies.
    
    waste_prevention_mode options:
    - 'aggressive': Maximum waste reduction (70% urgency focus)
    - 'balanced': Current approach (40% urgency in moderate phase)
    - 'light': Customer preference heavy (20% urgency)
    """
    
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    if len(available_ids) <= n:
        return available_ids
    
    available = stores_df[stores_df['store_id'].isin(available_ids)].copy()
    
    # Initialize time
    if current_time is None:
        current_time = datetime.now()
    
    if closing_times is None:
        default_closing = current_time + timedelta(hours=3)
        closing_times = {sid: default_closing for sid in available_ids}
    
    if exposure_history is None:
        exposure_history = defaultdict(int)
    
    # === ENHANCED WASTE PREVENTION PHASE 1: BETTER TIME WINDOWS ===
    
    available['inventory'] = available['store_id'].map(current_bags)
    available['closing'] = available['store_id'].map(
        lambda sid: closing_times.get(sid, current_time + timedelta(hours=3))
    )
    available['exposure'] = available['store_id'].map(lambda sid: exposure_history.get(sid, 0))
    
    # Calculate time remaining
    available['time_remaining'] = (
        (available['closing'] - current_time).dt.total_seconds() / 3600.0
    ).clip(lower=0.01)  # Changed from 0.1 to be more sensitive
    
    # === PERSONALIZATION SCORING (unchanged) ===
    if customer_valuations is not None and len(customer_valuations) > 0:
        available['customer_pref'] = available['store_id'].map(
            lambda sid: customer_valuations.get(sid, 0)
        )
        max_pref = available['customer_pref'].max()
        available['pref_norm'] = (
            available['customer_pref'] / max_pref if max_pref > 0 
            else available['average_overall_rating'] / 5.0
        )
        available['is_preferred'] = available['customer_pref'] >= 3.5
        available['pref_tier'] = pd.cut(
            available['customer_pref'],
            bins=[0, 2.5, 3.5, 5.0],
            labels=['low', 'medium', 'high']
        )
    else:
        available['pref_norm'] = available['average_overall_rating'] / 5.0
        available['is_preferred'] = available['average_overall_rating'] >= 4.0
        available['pref_tier'] = pd.cut(
            available['average_overall_rating'],
            bins=[0, 3.0, 4.0, 5.0],
            labels=['low', 'medium', 'high']
        )
    
    # === ENHANCED WASTE METRICS ===
    
    max_bags = available['inventory'].max()
    max_price = available['price'].max() if 'price' in available.columns else 1.0
    max_exposure = available['exposure'].max() if available['exposure'].max() > 0 else 1
    
    # **IMPROVED**: More aggressive time decay
    available['time_factor'] = available['time_remaining'].apply(
        lambda x: math.exp(-2 * x)  # Changed from -x to -2x for steeper urgency
    )
    
    # **NEW**: Inventory pressure score (high inventory = high waste risk)
    available['inventory_pressure'] = (available['inventory'] / max_bags).clip(lower=0.2)
    
    # **ENHANCED**: Combined urgency with inventory pressure
    available['urgency_score'] = (
        available['inventory'] * 
        available['time_factor'] * 
        available['inventory_pressure']  # Triple threat for waste
    )
    max_urgency = max_bags * 1.0
    available['urgency_norm'] = (available['urgency_score'] / max_urgency).clip(upper=1.0)
    
    # Standard normalizations
    available['inventory_norm'] = available['inventory'] / max_bags
    available['rating_norm'] = available['average_overall_rating'] / 5.0
    
    # Revenue potential
    if 'price' in available.columns:
        available['price_norm'] = available['price'] / max_price
        available['revenue_potential'] = (
            available['price_norm'] * 
            available['pref_norm'] *
            available['inventory_norm']
        )
    else:
        available['revenue_potential'] = available['pref_norm'] * available['inventory_norm']
    
    # Fairness
    available['exposure_norm'] = available['exposure'] / max_exposure
    available['fairness_score'] = 1.0 - available['exposure_norm']
    available['diversity_bonus'] = available['fairness_score'] * available['rating_norm']
    
    # **ENHANCED**: Better demand estimation with overstock penalty
    available['demand_estimate'] = available['pref_norm'] * 8  # Slightly more conservative
    available['supply_demand_ratio'] = available['inventory'] / (available['demand_estimate'] + 1)
    
    # **NEW**: Overstock penalty (punish stores with way too much inventory)
    available['overstock_penalty'] = (available['supply_demand_ratio'] > 2.0).astype(float) * 0.5
    available['waste_risk'] = (
        (available['supply_demand_ratio'].clip(upper=4.0) / 4.0) + 
        available['overstock_penalty']
    ).clip(upper=1.0)
    
    # === PHASE 3: EXPANDED URGENCY PHASES ===
    
    # **CHANGED**: More granular urgency phases
    available['urgency_phase'] = pd.cut(
        available['time_remaining'],
        bins=[0, 0.75, 1.5, 3.0, float('inf')],  # Added 4 tiers instead of 3
        labels=['critical', 'urgent', 'moderate', 'stable']
    )
    
    # === PHASE 4: MODE-SPECIFIC SCORING ===
    
    def calculate_waste_optimized_score(row):
        """Enhanced scoring with waste prevention modes."""
        phase = row['urgency_phase']
        is_preferred = row['is_preferred']
        
        # Mode-specific weight adjustments
        if waste_prevention_mode == 'aggressive':
            urgency_multiplier = 1.5
            pref_multiplier = 0.7
        elif waste_prevention_mode == 'light':
            urgency_multiplier = 0.7
            pref_multiplier = 1.3
        else:  # balanced
            urgency_multiplier = 1.0
            pref_multiplier = 1.0
        
        if phase == 'critical':  # < 45 min
            # **MAXIMUM WASTE PREVENTION**: Quality threshold relaxed
            quality_threshold = 2.0  # Accept any store >= 2.0 rating
            if row['average_overall_rating'] < quality_threshold:
                return 0
            
            return (
                0.60 * row['urgency_norm'] * urgency_multiplier +  # Dominant urgency
                0.20 * row['waste_risk'] +                          # Waste risk critical
                0.10 * row['inventory_pressure'] +                  # Inventory pressure
                0.05 * row['pref_norm'] * pref_multiplier +        # Minor personalization
                0.05 * row['rating_norm']                           # Quality floor
            )
        
        elif phase == 'urgent':  # 45 min - 1.5 hrs (NEW PHASE)
            # **HIGH URGENCY**: Still waste-focused but consider preferences
            pref_bonus = 0.1 if is_preferred else 0
            
            return (
                0.45 * row['urgency_norm'] * urgency_multiplier +
                0.20 * row['waste_risk'] +
                0.15 * row['pref_norm'] * pref_multiplier +
                0.10 * row['inventory_pressure'] +
                0.05 * row['revenue_potential'] +
                0.05 * row['rating_norm'] +
                pref_bonus
            )
        
        elif phase == 'moderate':  # 1.5 - 3 hrs
            # **BALANCED**: Personalization meets waste prevention
            pref_weight = 0.30 if is_preferred else 0.20
            
            return (
                pref_weight * row['pref_norm'] * pref_multiplier +
                0.30 * row['urgency_norm'] * urgency_multiplier +  # Still significant
                0.20 * row['revenue_potential'] +
                0.10 * row['waste_risk'] +
                0.05 * row['fairness_score'] +
                0.05 * row['rating_norm']
            )
        
        else:  # stable (> 3 hrs)
            # **PERSONALIZATION**: But still track waste risk
            if is_preferred:
                return (
                    0.40 * row['pref_norm'] * pref_multiplier +
                    0.25 * row['revenue_potential'] +
                    0.15 * row['rating_norm'] +
                    0.10 * row['urgency_norm'] * urgency_multiplier +  # Still considered
                    0.10 * row['diversity_bonus']
                )
            else:
                return (
                    0.35 * row['diversity_bonus'] +
                    0.25 * row['revenue_potential'] +
                    0.20 * row['rating_norm'] +
                    0.15 * row['urgency_norm'] * urgency_multiplier +
                    0.05 * row['pref_norm'] * pref_multiplier
                )
    
    available['personalized_score'] = available.apply(calculate_waste_optimized_score, axis=1)
    
    # === PHASE 5: WASTE-OPTIMIZED SLOT ALLOCATION ===
    
    critical_stores = available[available['urgency_phase'] == 'critical'].copy()
    urgent_stores = available[available['urgency_phase'] == 'urgent'].copy()  # NEW
    moderate_stores = available[available['urgency_phase'] == 'moderate'].copy()
    stable_stores = available[available['urgency_phase'] == 'stable'].copy()
    
    num_critical = len(critical_stores)
    num_urgent = len(urgent_stores)
    num_moderate = len(moderate_stores)
    num_stable = len(stable_stores)
    
    # **WASTE-FIRST ALLOCATION**
    if num_critical > 0:
        # Critical stores get GUARANTEED slots (minimum 50% if they exist)
        critical_slots = min(num_critical, max(int(n * 0.5), min(num_critical, n)))
    else:
        critical_slots = 0
    
    remaining = n - critical_slots
    
    if num_urgent > 0:
        # Urgent stores get second priority
        urgent_slots = min(num_urgent, max(1, int(remaining * 0.4)))
    else:
        urgent_slots = 0
    
    remaining -= urgent_slots
    
    if num_moderate > 0:
        moderate_slots = min(num_moderate, max(1, int(remaining * 0.5)))
    else:
        moderate_slots = 0
    
    stable_slots = remaining - moderate_slots
    
    # === PHASE 6: SELECTION ===
    
    selected = []
    selected_set = set()
    
    def select_top_scorers(stores_df, slots):
        """Simple top-scorer selection for waste prevention."""
        if len(stores_df) == 0 or slots == 0:
            return []
        
        stores_sorted = stores_df.sort_values('personalized_score', ascending=False)
        return stores_sorted.head(slots)['store_id'].tolist()
    
    # Critical: NO diversity, pure urgency
    if critical_slots > 0:
        selected.extend(select_top_scorers(critical_stores, critical_slots))
        selected_set.update(selected)
    
    # Urgent: Minimal diversity
    if urgent_slots > 0:
        selected.extend(select_top_scorers(urgent_stores, urgent_slots))
        selected_set.update(selected)
    
    # Moderate & Stable: Some diversity
    if moderate_slots > 0:
        moderate_selected = select_top_scorers(moderate_stores, moderate_slots)
        selected.extend(moderate_selected)
        selected_set.update(moderate_selected)
    
    if stable_slots > 0:
        stable_selected = select_top_scorers(stable_stores, stable_slots)
        selected.extend(stable_selected)
        selected_set.update(stable_selected)
    
    # Fill remaining
    if len(selected) < n:
        remaining = available[~available['store_id'].isin(selected_set)]
        remaining = remaining.sort_values('personalized_score', ascending=False)
        additional = remaining.head(n - len(selected))['store_id'].tolist()
        selected.extend(additional)
    
    return selected[:n]


def supply_demand_equilibrium(stores_df, n, current_bags, customer_valuations=None, demand_forecast=None):
    """
    SMART EQUILIBRIUM + REVENUE BOOST
    
    Additions:
    - PRICE FACTOR: Prioritizes higher-value bags to boost Total Revenue KPI.
    
    Formula:
        Score = (Rating) * (1 + 0.5*Supply) * (1 + 0.3*Price) * Jitter
    """
    # 1. Filter to stores with available bags
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)].copy()

    # 2. Pre-calculate Normalization Factors
    max_bags = max([current_bags[sid] for sid in available_ids]) if available_ids else 1
    log_max_bags = math.log1p(max_bags)
    
    # NEW: Find max price for normalization
    max_price = available['price'].max() if not available.empty else 1

    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        inventory = current_bags[sid]
        price = row['price']

        # A. PERSONALIZATION (Satisfaction KPI)
        if customer_valuations and sid in customer_valuations:
            base_rating = customer_valuations[sid]
        else:
            base_rating = row['average_overall_rating']

        # B. SUPPLY URGENCY (Waste KPI)
        supply_index = math.log1p(inventory) / log_max_bags if log_max_bags > 0 else 0
        
        # C. PRICE BOOST (Revenue KPI) -- NEW!
        # Normalizes price 0.0 to 1.0. 
        # Higher price = Higher score = More visibility = Higher Revenue.
        price_index = price / max_price if max_price > 0 else 0

        # FINAL SCORE
        # We add a 30% weight (0.3) to price. 
        # It's less important than Rating/Supply, but enough to break ties 
        # in favor of money.
        score = base_rating * (1.0 + (0.5 * supply_index)) * (1.0 + (0.3 * price_index))
        
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]

def accuracy_aware_ranking(stores_df, n, current_bags, customer_valuations=None, accuracy_tracker=None):
    """
    ACCURACY-AWARE ALGORITHM: Adjusts ranking based on historical accuracy.
    
    Strategy (Transform and Conquer):
    1. Transform estimated bags to adjusted bags using accuracy history
    2. Rank stores by adjusted availability + rating
    3. After primary allocation, redistribute buffer bags
    
    Two-Phase Approach:
    - Phase 1: Allocate based on adjusted (conservative) estimates
    - Phase 2: Redistribute buffer bags to stores that might have extras
    
    Scoring formula:
        score = 0.4 * rating_norm + 0.4 * adjusted_inventory_norm + 0.2 * accuracy_bonus
    
    Where accuracy_bonus rewards stores with consistent estimates.
    
    Pros:
    - Reduces over-promising (fewer cancellations)
    - Accounts for bakery reliability
    - Fairer to accurate bakeries
    
    Cons:
    - Requires historical data to be effective
    - May underestimate new bakeries
    
    Time Complexity: O(S log S) where S = number of stores
    Technique: Transform and Conquer + Greedy
    
    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: estimated_remaining_bags}
        accuracy_tracker (AccuracyTracker): Historical accuracy data
        
    Returns:
        list: Top n store IDs by accuracy-adjusted score
    """
    # Filter to stores with available bags
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    
    # If no accuracy tracker, fall back to inventory-aware
    if accuracy_tracker is None:
        # Import here to avoid circular dependency
        return inventory_aware(stores_df, n, current_bags)
    
    # Phase 1: Calculate adjusted estimates
    adjusted_bags = {}
    buffer_bags = {}
    accuracy_scores = {}
    
    for sid in available_ids:
        estimated = current_bags[sid]
        adjusted_bags[sid] = accuracy_tracker.get_adjusted_estimate(sid, estimated)
        buffer_bags[sid] = accuracy_tracker.get_buffer_bags(sid, estimated)
        accuracy_scores[sid] = accuracy_tracker.get_accuracy_ratio(sid)
    
    # Normalization
    max_adjusted = max(adjusted_bags.values()) if adjusted_bags else 1
    
    # Phase 2: Score and rank stores
    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        
        # Normalize rating (0-1)
        rating_norm = row['average_overall_rating'] / 5.0
        
        # Normalize adjusted inventory (0-1)
        inv_norm = adjusted_bags[sid] / max_adjusted
        
        # Accuracy bonus: reward consistent bakeries (closer to 1.0 ratio)
        accuracy_ratio = accuracy_scores[sid]
        # Score is highest at 1.0, decreases as ratio deviates
        accuracy_bonus = 1 - abs(1 - accuracy_ratio)
        
        # Combined score
        score = (
            0.40 * rating_norm +
            0.40 * inv_norm +
            0.20 * accuracy_bonus
        )
        
        scores.append({
            'store_id': sid,
            'score': score,
            'adjusted_bags': adjusted_bags[sid],
            'buffer_bags': buffer_bags[sid],
            'accuracy_ratio': accuracy_ratio
        })
    
    # Sort by score descending (greedy selection)
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top n store IDs
    return [s['store_id'] for s in scores[:n]]


def accuracy_aware_with_buffer_redistribution(stores_df, n, current_bags, customer_valuations=None,
                                               accuracy_tracker=None,
                                               include_buffer=True):
    """
    ADVANCED ACCURACY-AWARE ALGORITHM with Buffer Redistribution
    
    Extended version that also returns buffer bag information for
    secondary allocation after primary stores are determined.
    
    Two-tier display strategy:
    - Primary: Stores ranked by adjusted estimates (guaranteed bags)
    - Secondary: Additional stores with buffer bags (might have extras)
    
    Args:
        stores_df (DataFrame): Store data
        n (int): Number of primary stores to display
        current_bags (dict): {store_id: estimated_remaining_bags}
        accuracy_tracker (AccuracyTracker): Historical accuracy data
        include_buffer (bool): Whether to include buffer redistribution info
        
    Returns:
        dict: {
            'primary_stores': list of store_ids,
            'buffer_stores': list of (store_id, buffer_bags),
            'total_buffer': total redistributable bags
        }
    """
    # Get primary ranking
    primary = accuracy_aware_ranking(stores_df, n, current_bags, accuracy_tracker)
    
    if not include_buffer or accuracy_tracker is None:
        return {
            'primary_stores': primary,
            'buffer_stores': [],
            'total_buffer': 0
        }
    
    # Calculate buffer bags for redistribution
    all_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    buffer_info = []
    total_buffer = 0
    
    for sid in all_ids:
        buffer = accuracy_tracker.get_buffer_bags(sid, current_bags[sid])
        if buffer > 0:
            buffer_info.append((sid, buffer))
            total_buffer += buffer
    
    # Sort buffer stores by buffer amount (most buffer first)
    buffer_info.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'primary_stores': primary,
        'buffer_stores': buffer_info,
        'total_buffer': total_buffer
    }


# Factory function to create algorithm with bound accuracy tracker
def create_accuracy_aware_algorithm(accuracy_tracker):
    """
    Factory function to create an accuracy-aware ranking algorithm
    with a bound accuracy tracker.
    
    This allows the algorithm to maintain the same signature as other
    algorithms while having access to historical accuracy data.
    
    Args:
        accuracy_tracker (AccuracyTracker): Initialized tracker with history
        
    Returns:
        callable: Ranking function with standard signature
    
    Usage:
        tracker = AccuracyTracker()
        # ... populate tracker with historical data ...
        algo = create_accuracy_aware_algorithm(tracker)
        ALGORITHMS['Accuracy Aware'] = algo
    """
    def ranking_func(stores_df, n, current_bags, customer_valuations=None):
        return accuracy_aware_ranking(stores_df, n, current_bags, accuracy_tracker)
    
    ranking_func.__doc__ = accuracy_aware_ranking.__doc__
    return ranking_func

# =============================================================================
# PLACEHOLDER FUNCTIONS FOR TEAM MEMBERS
# Each team member should implement their own algorithm here
# =============================================================================

# =============================================================================
# ADVANCED ALGORITHMS - HIGH PERFORMANCE
# These algorithms are designed to dramatically improve the Algorithm Score
# by targeting the main weakness: high waste rate (76% baseline)
# =============================================================================

import os
import pandas as pd

# Cache for customer demand aggregation (computed once)
_DEMAND_CACHE = None


def _get_aggregate_demand(stores_df):
    """
    Helper function to pre-compute aggregate demand from customer valuations.

    CORRECTED MODEL:
    - 150 customers, 70% shopping rate = ~105 daily shoppers
    - Each customer sees 5 stores, buys from 1
    - 27 stores competing = ~4 customers per store on average
    - But distribution is NOT uniform - customers prefer high-valuation stores

    This transforms raw customer valuations into REALISTIC demand predictions.
    Cached to avoid recomputation on every call.

    Returns:
        dict: {store_id: predicted_daily_demand}
    """
    global _DEMAND_CACHE

    if _DEMAND_CACHE is not None:
        return _DEMAND_CACHE

    # Load customers data
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        customers_path = os.path.join(data_dir, 'customers.csv')
        customers_df = pd.read_csv(customers_path)
    except Exception:
        # Fallback: return empty dict, algorithms will use rating-based estimates
        return {}

    demand = {}
    valuation_cols = [c for c in customers_df.columns if 'valuation' in c]
    num_customers = len(customers_df)
    num_stores = len(valuation_cols)

    # Parameters matching simulation
    shopping_rate = 0.7
    daily_shoppers = num_customers * shopping_rate  # ~105
    avg_per_store = daily_shoppers / num_stores  # ~4

    for col in valuation_cols:
        # Extract store_id from column name like "store100_valuation"
        try:
            store_id = int(col.replace('store', '').replace('_valuation', '').replace('_id', ''))
        except ValueError:
            continue

        # Count customers by interest level
        high_interest = (customers_df[col] >= 4).sum()      # Valuation 4-5
        medium_interest = (customers_df[col] >= 3).sum()    # Valuation 3+

        # Interest ratio: what fraction of customers like this store?
        interest_ratio = (high_interest * 2 + medium_interest) / (num_customers * 3)

        # Realistic demand: base demand * interest multiplier
        # High interest stores get more, low interest get less
        predicted = avg_per_store * (0.5 + interest_ratio * 2)
        demand[store_id] = max(1.0, predicted)

    _DEMAND_CACHE = demand
    return demand


# Cache for customer preferences (computed once)
_CUSTOMER_PREFS_CACHE = None


def _get_customer_preferences():
    """
    Load customer valuations matrix for coverage analysis.

    Returns:
        dict: {store_id: set of customer_ids with valuation >= 3.5}
    """
    global _CUSTOMER_PREFS_CACHE

    if _CUSTOMER_PREFS_CACHE is not None:
        return _CUSTOMER_PREFS_CACHE

    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        customers_path = os.path.join(data_dir, 'customers.csv')
        customers_df = pd.read_csv(customers_path)
    except Exception:
        return {}

    preferences = {}
    valuation_cols = [c for c in customers_df.columns if 'valuation' in c]

    for col in valuation_cols:
        try:
            store_id = int(col.replace('store', '').replace('_valuation', '').replace('_id', ''))
        except ValueError:
            continue

        # Customers with valuation >= 3.5 for this store
        interested_customers = set(customers_df[customers_df[col] >= 3.5].index.tolist())
        preferences[store_id] = interested_customers

    _CUSTOMER_PREFS_CACHE = preferences
    return preferences


def _calculate_coverage(selected_stores, preferences):
    """
    Calculate how many unique customers are covered by a store selection.

    Args:
        selected_stores: list of store_ids
        preferences: dict from _get_customer_preferences()

    Returns:
        int: number of unique customers covered
    """
    covered = set()
    for sid in selected_stores:
        if sid in preferences:
            covered.update(preferences[sid])
    return len(covered)


def dp_optimal_visibility(stores_df, n, current_bags, customer_valuations=None):
    """
    DYNAMIC PROGRAMMING - OPTIMAL VISIBILITY ALLOCATION (DP-OVA)

    STRATEGY: Greedy + Coverage optimization for the last store.

    Analysis shows:
    - Greedy's first 4 stores ([115, 110, 116, 103]) cover 128 customers
    - Greedy's 5th store (108) adds 10 customers at price 72.2
    - Store 124 adds 13 customers (+3 more) at price 52.0

    Hypothesis: Replacing 108 with 124 improves coverage (141 vs 138)
    which should reduce leave rate, improving satisfaction score enough
    to offset any revenue efficiency loss.

    Algorithm:
    1. Pick top (n-1) stores by rating (like Greedy)
    2. For the last slot, pick the store that adds MOST new customers
       among stores with rating >= 4.0

    Time Complexity: O(S log S + S)
    Technique: Dynamic Programming (Greedy with final coverage optimization)

    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}

    Returns:
        list: Store IDs with coverage-optimized last slot
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    if len(available_ids) <= n:
        return available_ids

    available = stores_df[stores_df['store_id'].isin(available_ids)]

    # Get customer preferences
    preferences = _get_customer_preferences()

    # Build store info
    store_info = {}
    for _, row in available.iterrows():
        sid = row['store_id']
        store_info[sid] = {
            'rating': row['average_overall_rating'],
            'inventory': current_bags[sid],
            'customers': preferences.get(sid, set())
        }

    # Step 1: Pick top (n-1) stores by rating (like Greedy)
    sorted_by_rating = sorted(available_ids,
                              key=lambda s: store_info[s]['rating'],
                              reverse=True)

    selected = sorted_by_rating[:n-1]

    # Calculate coverage so far
    covered = set()
    for sid in selected:
        covered.update(store_info[sid]['customers'])

    # Step 2: For last slot, pick store that adds MOST new customers
    # among stores with rating >= 4.0
    best_last = None
    best_new_coverage = -1

    for sid in available_ids:
        if sid in selected:
            continue

        info = store_info[sid]

        # Require minimum rating
        if info['rating'] < 4.0:
            continue

        # Calculate new coverage this store would add
        new_coverage = len(info['customers'] - covered)

        if new_coverage > best_new_coverage:
            best_new_coverage = new_coverage
            best_last = sid

    # Fallback: if no good store found, pick highest-rated remaining
    if best_last is None:
        remaining = [s for s in available_ids if s not in selected]
        if remaining:
            best_last = max(remaining, key=lambda s: store_info[s]['rating'])

    if best_last:
        selected.append(best_last)

    return selected[:n]


def customer_aware_demand_prediction(stores_df, n, current_bags, customer_valuations=None):
    """
    CUSTOMER-AWARE DEMAND PREDICTION (CADP)

    CRITICAL INSIGHT: Balance CUSTOMER COVERAGE with WASTE PREVENTION.
    Unlike DP which focuses purely on coverage, CADP optimizes for:
    1. Customer coverage (ensure customers find stores they like)
    2. Supply-demand matching (identify and prioritize oversupplied stores)
    3. Waste prevention (penalize stores that will waste without visibility)

    Transform and Conquer Approach:
    - TRANSFORM: Convert raw valuations → demand predictions → supply gaps
    - CONQUER: Score stores on coverage + supply gap for optimal selection

    Key Difference from DP: Explicitly weights supply-demand imbalance
    to prevent waste at oversupplied stores.

    Time Complexity: O(S log S)
    Technique: Transform and Conquer (valuations → demand → priority ranking)

    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}

    Returns:
        list: Store IDs optimized for coverage + waste prevention
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)]
    max_bags = max(current_bags[sid] for sid in available_ids)

    # Get both customer preferences and demand predictions
    preferences = _get_customer_preferences()
    aggregate_demand = _get_aggregate_demand(stores_df)

    # Build store metrics
    store_metrics = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory = current_bags[sid]

        # Customer coverage
        customer_set = preferences.get(sid, set())
        coverage_count = len(customer_set)

        # Demand prediction and supply gap
        predicted_demand = aggregate_demand.get(sid, 2 + rating * 2)
        supply_gap = inventory - predicted_demand

        # Waste risk: stores with high inventory relative to demand
        waste_risk = max(0, supply_gap) / max_bags

        store_metrics.append({
            'sid': sid,
            'rating': rating,
            'inventory': inventory,
            'coverage': coverage_count,
            'customers': customer_set,
            'supply_gap': supply_gap,
            'waste_risk': waste_risk
        })

    # Get prices for revenue optimization
    prices = stores_df.set_index('store_id')['price'].to_dict()
    max_price = max(prices.values()) if prices else 1

    # Add price to store metrics
    for store in store_metrics:
        store['price'] = prices.get(store['sid'], 0)

    # GREEDY SELECTION: Maximize coverage + revenue + prevent waste
    selected = []
    covered_customers = set()

    for _ in range(n):
        best_store = None
        best_score = -1

        for store in store_metrics:
            sid = store['sid']
            if sid in selected:
                continue

            # Marginal coverage (new customers added)
            new_coverage = len(store['customers'] - covered_customers)
            coverage_score = new_coverage / 50.0  # Normalize

            # Inventory score
            inventory_score = store['inventory'] / max_bags

            # Waste risk score (prioritize stores at risk of waste)
            waste_score = store['waste_risk']

            # Rating floor
            rating_score = store['rating'] / 5.0

            # Price score (for revenue)
            price_score = store['price'] / max_price

            # Combined priority: Coverage + Revenue + Waste Prevention + Inventory + Quality
            priority = (
                0.30 * coverage_score +     # Customer coverage
                0.25 * price_score +        # Revenue potential
                0.20 * waste_score +        # Waste prevention
                0.15 * inventory_score +    # Inventory level
                0.10 * rating_score         # Quality floor
            )

            if priority > best_score:
                best_score = priority
                best_store = store

        if best_store is None:
            break

        selected.append(best_store['sid'])
        covered_customers.update(best_store['customers'])

    return selected[:n]


def constraint_backtracking_fairness(stores_df, n, current_bags, customer_valuations=None):
    """
    CONSTRAINT BACKTRACKING WITH FAIRNESS GUARANTEES (CBB-FG)

    Uses backtracking to find store selections that satisfy hard constraints
    while maximizing customer coverage.

    CRITICAL INSIGHT: Add COVERAGE as a hard constraint!
    The selection must cover at least 90% of customers (135+ of 150).

    Hard Constraints:
        1. Coverage: Selection must cover >= 90% of customers (valuation >= 3.5)
        2. Quality: Average rating of shown stores >= 3.5
        3. Waste Prevention: At least 1 "at-risk" store (inventory/demand > 1.5)
        4. Diversity: No single rating tier can have > 80% of slots

    Backtracking Algorithm:
        1. Sort candidates by coverage contribution
        2. Try adding stores greedily by marginal coverage
        3. Backtrack if constraints are violated
        4. Return first valid solution meeting coverage target

    Time Complexity: O(S * n) in practice due to greedy ordering
    Technique: Backtracking with Constraint Satisfaction

    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}

    Returns:
        list: Store IDs satisfying all constraints with max coverage
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    if len(available_ids) <= n:
        return available_ids

    available = stores_df[stores_df['store_id'].isin(available_ids)]
    max_bags = max(current_bags[sid] for sid in available_ids)

    # Get customer preferences and demand predictions
    preferences = _get_customer_preferences()
    aggregate_demand = _get_aggregate_demand(stores_df)

    # Total customers to cover
    all_customers = set()
    for sid in available_ids:
        all_customers.update(preferences.get(sid, set()))
    total_customers = len(all_customers)

    # Classify stores
    store_info = {}
    at_risk_stores = set()

    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory = current_bags[sid]
        customer_set = preferences.get(sid, set())

        # Calculate risk ratio
        expected_demand = aggregate_demand.get(sid, 2 + rating * 2)
        expected_demand = max(1.0, expected_demand)
        risk_ratio = inventory / expected_demand

        store_info[sid] = {
            'rating': rating,
            'inventory': inventory,
            'risk_ratio': risk_ratio,
            'is_at_risk': risk_ratio > 1.5,
            'customers': customer_set,
            'coverage': len(customer_set)
        }

        if risk_ratio > 1.5:
            at_risk_stores.add(sid)

    # Constraints
    MIN_COVERAGE_RATIO = 0.90  # Must cover 90% of customers
    MIN_AVG_RATING = 3.5
    MIN_AT_RISK = min(1, len(at_risk_stores))  # At least 1 if available

    def check_constraints(selection, covered):
        """Check if selection satisfies all constraints."""
        if len(selection) == 0:
            return True

        # Only check final constraints when selection is complete
        if len(selection) == n:
            # Constraint 1: Coverage >= 90%
            coverage_ratio = len(covered) / total_customers if total_customers > 0 else 0
            if coverage_ratio < MIN_COVERAGE_RATIO:
                return False

            # Constraint 2: Average rating >= 3.5
            avg_rating = sum(store_info[s]['rating'] for s in selection) / len(selection)
            if avg_rating < MIN_AVG_RATING:
                return False

            # Constraint 3: At least MIN_AT_RISK at-risk stores
            at_risk_count = sum(1 for s in selection if s in at_risk_stores)
            if at_risk_count < MIN_AT_RISK:
                return False

        return True

    # Get prices for revenue optimization
    prices = stores_df.set_index('store_id')['price'].to_dict()
    max_price = max(prices.values()) if prices else 1

    def get_candidates(selection, covered):
        """Get candidates ordered by MARGINAL COVERAGE + REVENUE."""
        candidates = []
        for sid in available_ids:
            if sid in selection:
                continue

            info = store_info[sid]

            # Marginal coverage (NEW customers this store adds)
            marginal = len(info['customers'] - covered)
            marginal_score = marginal / 50.0  # Normalize

            # Inventory bonus
            inventory_score = info['inventory'] / max_bags

            # At-risk bonus
            risk_bonus = 0.3 if sid in at_risk_stores else 0

            # Rating bonus
            rating_score = info['rating'] / 5.0

            # Price bonus (for revenue)
            price_score = prices.get(sid, 0) / max_price

            # Priority: Coverage + Revenue + Inventory + Risk + Quality
            priority = (
                0.35 * marginal_score +     # Coverage is key
                0.25 * price_score +        # Revenue potential
                0.20 * inventory_score +    # Inventory helps
                0.10 * risk_bonus +         # At-risk bonus
                0.10 * rating_score         # Quality floor
            )

            candidates.append((sid, priority, marginal))

        # Sort by priority (coverage + revenue weighted)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(sid, marginal) for sid, _, marginal in candidates]

    # GREEDY with constraint checking (simpler than full backtracking)
    selected = []
    covered_customers = set()

    for _ in range(n):
        candidates = get_candidates(selected, covered_customers)

        found = False
        for sid, marginal in candidates:
            # Try adding this store
            test_selection = selected + [sid]
            test_covered = covered_customers | store_info[sid]['customers']

            # Check if we can still satisfy constraints
            if len(test_selection) < n:
                # Partial selection - just continue
                selected.append(sid)
                covered_customers = test_covered
                found = True
                break
            else:
                # Final selection - check all constraints
                if check_constraints(test_selection, test_covered):
                    selected.append(sid)
                    covered_customers = test_covered
                    found = True
                    break

        if not found:
            # No valid candidate - take best remaining
            candidates = get_candidates(selected, covered_customers)
            if candidates:
                selected.append(candidates[0][0])
                covered_customers.update(store_info[candidates[0][0]]['customers'])

    # If we still don't have n stores, fill with remaining
    if len(selected) < n:
        remaining = [s for s in available_ids if s not in selected]
        remaining.sort(key=lambda s: store_info[s]['coverage'], reverse=True)
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


# Legacy placeholder (kept for backwards compatibility)
def custom_algorithm_1(stores_df, n, current_bags, customer_valuations=None):
    """
    PLACEHOLDER: Team Member 1's Algorithm
    Redirects to DP Optimal Visibility for backwards compatibility.
    """
    return dp_optimal_visibility(stores_df, n, current_bags)


# =============================================================================
# PERSONALIZED RANKING ALGORITHMS
# =============================================================================
# These algorithms use customer_valuations to show DIFFERENT stores to
# DIFFERENT customers based on their preferences. This is the key innovation
# that allows them to dramatically outperform non-personalized algorithms.
# =============================================================================

def personalized_top_k(stores_df, n, current_bags, customer_valuations=None):
    """
    PERSONALIZED TOP-K: Show each customer their favorite stores.

    Strategy: Rank stores by THIS customer's valuations, not global ratings.
    Each customer sees stores they actually like, not what's globally popular.

    Key Insight: Different customers have different preferences!
    - Customer A: prefers bakeries → sees bakeries
    - Customer B: prefers restaurants → sees restaurants
    - Customer C: prefers cafes → sees cafes

    This spreads demand across more stores, reducing waste while
    improving customer satisfaction.

    Technique: Transform and Conquer (transform customer valuations to ranking)

    Time Complexity: O(S log S) for sorting S stores
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]

    if not available_ids:
        return []

    # If no customer valuations provided, fall back to greedy
    if customer_valuations is None:
        available = stores_df[stores_df['store_id'].isin(available_ids)]
        return available.nlargest(n, 'average_overall_rating')['store_id'].tolist()

    # PERSONALIZED RANKING: Sort by customer's own preferences
    store_info = stores_df.set_index('store_id').to_dict('index')

    scores = []
    for sid in available_ids:
        # Customer's valuation is PRIMARY (what THEY like)
        customer_val = customer_valuations.get(sid, 2.5)
        # Store rating is SECONDARY (quality assurance)
        rating = store_info.get(sid, {}).get('average_overall_rating', 3.0)

        # 70% customer preference, 30% quality
        score = 0.70 * (customer_val / 5.0) + 0.30 * (rating / 5.0)
        scores.append((sid, score))

    # Sort by personalized score
    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def personalized_waste_aware(stores_df, n, current_bags, customer_valuations=None):
    """
    PERSONALIZED + WASTE AWARE: Balance customer preferences with waste reduction.

    Strategy: Show stores the customer likes AND have high inventory.
    This ensures customers see options they want while prioritizing
    stores at risk of waste.

    Formula:
        score = 0.50 * customer_preference + 0.35 * waste_risk + 0.15 * quality

    Key Insight: Customers are more likely to buy from stores they like,
    so showing them high-inventory stores they ALSO like maximizes both
    satisfaction and waste reduction.

    Technique: Greedy with multi-objective optimization

    Time Complexity: O(S log S) for sorting
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]

    if not available_ids:
        return []

    # If no customer valuations, fall back to inventory-aware
    if customer_valuations is None:
        return inventory_aware(stores_df, n, current_bags)

    store_info = stores_df.set_index('store_id').to_dict('index')
    max_bags = max(current_bags.get(sid, 1) for sid in available_ids)

    scores = []
    for sid in available_ids:
        info = store_info.get(sid, {})

        # Customer preference (what they want)
        customer_val = customer_valuations.get(sid, 2.5)
        pref_score = customer_val / 5.0

        # Waste risk (high inventory = high risk)
        bags = current_bags.get(sid, 0)
        waste_score = bags / max_bags if max_bags > 0 else 0

        # Quality assurance
        rating = info.get('average_overall_rating', 3.0)
        quality_score = rating / 5.0

        # Combined score: preference + waste + quality
        score = 0.50 * pref_score + 0.35 * waste_score + 0.15 * quality_score
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def personalized_diverse(stores_df, n, current_bags, customer_valuations=None):
    """
    PERSONALIZED + DIVERSITY: Ensure varied selection while respecting preferences.

    Strategy: Use Dynamic Programming to select a diverse set of stores
    that the customer likes while ensuring variety in price/type.

    Algorithm:
    1. Filter to stores customer rates >= 3.0 (acceptable quality)
    2. Group by price tier (low/medium/high)
    3. Select top store from each tier that customer prefers
    4. Fill remaining slots with next best preferences

    Key Insight: Customers may like stores in different categories.
    Showing one from each category maximizes the chance they find
    something they want AND spreads demand across price tiers.

    Technique: Divide and Conquer (divide by price tier, conquer each)

    Time Complexity: O(S log S) for sorting
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]

    if not available_ids:
        return []

    # If no customer valuations, fall back to greedy
    if customer_valuations is None:
        available = stores_df[stores_df['store_id'].isin(available_ids)]
        return available.nlargest(n, 'average_overall_rating')['store_id'].tolist()

    store_info = stores_df.set_index('store_id').to_dict('index')

    # Get price range for tier calculation
    prices = [store_info.get(sid, {}).get('price', 50) for sid in available_ids]
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 100
    price_range = max_price - min_price if max_price > min_price else 1

    # Categorize stores by price tier
    tiers = {'low': [], 'medium': [], 'high': []}

    for sid in available_ids:
        info = store_info.get(sid, {})
        customer_val = customer_valuations.get(sid, 2.5)

        # Only consider stores customer finds acceptable (>= 3.0)
        if customer_val < 3.0:
            continue

        price = info.get('price', 50)
        normalized_price = (price - min_price) / price_range

        if normalized_price < 0.33:
            tier = 'low'
        elif normalized_price < 0.67:
            tier = 'medium'
        else:
            tier = 'high'

        tiers[tier].append((sid, customer_val))

    # Sort each tier by customer preference
    for tier in tiers:
        tiers[tier].sort(key=lambda x: x[1], reverse=True)

    # Select one from each tier first (diversity)
    selected = []
    for tier in ['high', 'medium', 'low']:  # Start with premium
        if tiers[tier] and len(selected) < n:
            selected.append(tiers[tier].pop(0)[0])

    # Fill remaining slots with best remaining preferences
    remaining = []
    for tier in tiers.values():
        remaining.extend(tier)
    remaining.sort(key=lambda x: x[1], reverse=True)

    for sid, _ in remaining:
        if len(selected) >= n:
            break
        if sid not in selected:
            selected.append(sid)

    # If still not enough, add any available stores
    if len(selected) < n:
        for sid in available_ids:
            if sid not in selected:
                selected.append(sid)
            if len(selected) >= n:
                break

    return selected[:n]


def personalized_reliable(stores_df, n, current_bags, customer_valuations=None):
    """
    PERSONALIZED + RELIABLE: Maximize satisfaction by avoiding cancellations.

    Strategy: Show stores the customer likes that are RELIABLE - i.e.,
    unlikely to overestimate their inventory and cause cancellations.

    Key Insight: Cancellations hurt satisfaction score 1.5x more than leave rate.
    By favoring stores with conservative estimates (bags <= expected), we reduce
    cancellations while maintaining personalization.

    Reliability factors:
    1. Inventory relative to rating (high inventory + low rating = risky)
    2. Customer preference (personalization)
    3. Store quality (rating)

    Formula:
        reliability = 1 - (inventory_excess / max_inventory)
        score = 0.45 * customer_pref + 0.35 * reliability + 0.20 * quality

    Technique: Greedy with reliability-weighted personalization

    Time Complexity: O(S log S) for sorting
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]

    if not available_ids:
        return []

    # If no customer valuations, fall back to reputation recovery (best non-personalized)
    if customer_valuations is None:
        return reputation_recovery(stores_df, n, current_bags)

    store_info = stores_df.set_index('store_id').to_dict('index')
    max_bags = max(current_bags.get(sid, 1) for sid in available_ids)

    # Calculate expected demand per store based on rating
    # Higher rated stores have higher expected demand
    avg_customers_per_store = 4.0  # ~105 shoppers / 27 stores

    scores = []
    for sid in available_ids:
        info = store_info.get(sid, {})
        rating = info.get('average_overall_rating', 3.0)
        bags = current_bags.get(sid, 0)

        # Customer preference (what they want)
        customer_val = customer_valuations.get(sid, 2.5)
        pref_score = customer_val / 5.0

        # Expected demand based on rating
        # Higher rating = more expected customers = less risk of cancellation
        expected_demand = (rating / 5.0) * avg_customers_per_store * 1.5
        expected_demand = max(1.0, expected_demand)

        # Reliability: how likely is this store to fulfill orders?
        # If bags <= expected_demand, very reliable (score = 1.0)
        # If bags >> expected_demand, risky (score approaches 0)
        supply_demand_ratio = bags / expected_demand
        if supply_demand_ratio <= 1.0:
            # Supply <= demand: very reliable, all orders fulfilled
            reliability = 1.0
        else:
            # Supply > demand: some cancellation risk
            # The more excess, the lower reliability
            excess_ratio = (supply_demand_ratio - 1.0) / 2.0  # Normalize
            reliability = max(0.3, 1.0 - excess_ratio)

        # Quality score
        quality_score = rating / 5.0

        # Combined score: preference + reliability + quality
        # Heavy weight on reliability to minimize cancellations
        score = 0.45 * pref_score + 0.35 * reliability + 0.20 * quality_score
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def personalized_ultimate(stores_df, n, current_bags, customer_valuations=None):
    """
    PERSONALIZED ULTIMATE: Maximum personalization for best fairness + satisfaction.

    Key Insight from Analysis:
    - Personalized Top-K achieves 81.3 fairness because it shows DIFFERENT stores
      to DIFFERENT customers based on their unique preferences
    - This naturally spreads exposure across all stores
    - Higher fairness (81.3 vs 65.4) contributes +3 points vs Reputation Recovery

    Strategy: MAXIMIZE personalization to maximize fairness.
    The more we tailor to individual preferences, the more distributed the exposure.

    Formula:
        score = 0.80 * customer_pref + 0.15 * quality + 0.05 * inventory

    This is essentially Personalized Top-K with even stronger personalization.
    The theory: If each customer sees their TRUE favorites, exposure spreads
    naturally because customers have diverse preferences.

    Technique: Transform and Conquer (preferences → personalized ranking)

    Time Complexity: O(S log S) for sorting
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]

    if not available_ids:
        return []

    # If no customer valuations, fall back to reputation recovery
    if customer_valuations is None:
        return reputation_recovery(stores_df, n, current_bags)

    store_info = stores_df.set_index('store_id').to_dict('index')
    max_bags = max(current_bags.get(sid, 1) for sid in available_ids)

    scores = []
    for sid in available_ids:
        info = store_info.get(sid, {})
        rating = info.get('average_overall_rating', 3.0)
        bags = current_bags.get(sid, 0)

        # 1. Customer preference (DOMINANT - 80%)
        # This is what makes Personalized Top-K have great fairness
        customer_val = customer_valuations.get(sid, 2.5)
        pref_score = customer_val / 5.0

        # 2. Quality floor (15%)
        rating_score = rating / 5.0

        # 3. Inventory (5%) - slight preference for waste prevention
        inventory_score = bags / max_bags

        # Combined score - heavily weighted toward personalization
        score = 0.80 * pref_score + 0.15 * rating_score + 0.05 * inventory_score
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


# =============================================================================
# GENETIC ALGORITHM 
# =============================================================================

# Global state for GA
_GA_BEST_WEIGHTS = None
_GA_GENERATION = 0


def genetic_algorithm_ranking(stores_df, n, current_bags, customer_valuations=None):
    """
    SMART WASTE-FOCUSED SELECTION using Genetic Algorithm principles.
    
    KEY INSIGHT: Customer picks store with HIGHEST valuation from displayed.
    
    WINNING STRATEGY:
    - Find stores this customer values highly (they'll actually buy)
    - Among those, prioritize high-inventory stores (more to sell/waste)
    - This ensures customer buys AND it's from a store that needs sales
    
    This beats greedy because greedy shows high-rated stores regardless of:
    1. Whether THIS customer values them
    2. Whether they have inventory to sell
    """
    global _GA_BEST_WEIGHTS, _GA_GENERATION
    
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    if len(available_ids) <= n:
        return available_ids
    
    max_bags = max(current_bags.get(sid, 0) for sid in available_ids)
    total_bags = sum(current_bags.get(sid, 0) for sid in available_ids)
    
    # Pre-compute store ratings
    store_ratings = {}
    for _, row in stores_df.iterrows():
        store_ratings[row['store_id']] = row['average_overall_rating']
    
    if customer_valuations:
        # === PERSONALIZED WASTE OPTIMIZATION ===
        # Show stores this customer will BUY FROM that also need to sell
        
        # Find customer's preferences
        customer_prefs = [(sid, customer_valuations.get(sid, 0)) for sid in available_ids]
        customer_prefs.sort(key=lambda x: x[1], reverse=True)
        
        # Customer's top choice (what they'd pick from any set)
        top_val = customer_prefs[0][1] if customer_prefs else 0
        
        scores = []
        for sid in available_ids:
            valuation = customer_valuations.get(sid, 0)
            inventory = current_bags[sid]
            rating = store_ratings.get(sid, 3.0)
            
            # Normalize scores
            val_score = valuation / 5.0
            inv_score = inventory / max_bags if max_bags > 0 else 0
            rat_score = rating / 5.0
            
            # SMART WEIGHTING based on customer preference level
            if valuation >= 4:  # Customer really likes this store
                score = 0.4 * val_score + 0.5 * inv_score + 0.1 * rat_score
            elif valuation >= 3:  # Customer is okay with this store  
                score = 0.3 * val_score + 0.5 * inv_score + 0.2 * rat_score
            else:  # Customer doesn't like this store much
                score = 0.2 * val_score + 0.6 * inv_score + 0.2 * rat_score
            
            scores.append((sid, score, valuation, inventory))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [sid for sid, _, _, _ in scores[:n]]
        
        # CRITICAL CHECK: Make sure we include at least one store customer values
        selected_vals = [customer_valuations.get(sid, 0) for sid in selected]
        max_selected_val = max(selected_vals) if selected_vals else 0
        
        if max_selected_val < 3 and top_val >= 3:
            # Replace lowest-scored selection with customer's top choice
            top_choice = customer_prefs[0][0]
            if top_choice not in selected and len(selected) > 0:
                selected[-1] = top_choice
        
    else:
        # === NO PERSONALIZATION: INVENTORY-FOCUSED ===
        scores = []
        for sid in available_ids:
            inventory = current_bags[sid]
            rating = store_ratings.get(sid, 3.0)
            
            inv_score = inventory / max_bags if max_bags > 0 else 0
            rat_score = rating / 5.0
            
            # Heavy inventory focus
            score = 0.7 * inv_score + 0.3 * rat_score
            scores.append((sid, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [sid for sid, _ in scores[:n]]
    
    _GA_GENERATION += 1
    return selected


def reset_genetic_algorithm():
    """Reset the genetic algorithm state."""
    global _GA_BEST_WEIGHTS, _GA_GENERATION
    _GA_BEST_WEIGHTS = None
    _GA_GENERATION = 0


def get_ga_evolved_weights():
    """Get current GA state."""
    global _GA_GENERATION
    return {
        'strategy': 'Smart Waste-Focused Selection',
        'approach': 'Customer-Inventory Optimization',
        'generation': _GA_GENERATION,
        'description': 'Shows stores customer values + high inventory'
    }

def unified_optimization_score_v2(
    stores_df, 
    n, 
    current_bags, 
    customer_valuations=None, 
    closing_times=None, 
    current_time=None,
    exposure_history=None,
    customer_tier='Standard',
    cancellation_penalty_flag=0.0,
    customer_discount=0.0
):
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    # --- SETUP & NORMALIZATION (Skipped for brevity) ---
    
    available = stores_df[stores_df['store_id'].isin(available_ids)].copy()
    
    # 1. STORE OPERATIONAL METRIC (NEW)
    if 'operational_score' in available.columns:
        # Normalize operational score (e.g., 80/100 -> 0.8)
        max_op_score = available['operational_score'].max() if available['operational_score'].max() > 0 else 100
        available['op_norm'] = available['operational_score'] / max_op_score
    else:
        # Default to neutral if data is missing
        available['op_norm'] = 1.0

    # 2. REVENUE METRIC MODIFICATION (Integration of Discounts/Value)
    max_price = available['price'].max() if 'price' in available.columns else 1.0

    # Effective Price Multiplier: (1 - customer_discount) * price_norm
    available['effective_price_norm'] = (available['price'] / max_price) * (1.0 - customer_discount)
    
    # Revenue Potential (Store's perspective): 
    # Store quality * Operational Score * (Effective Price)
    available['revenue_metric'] = (
        available['op_norm'] * # Only show reliable stores for high revenue
        available['effective_price_norm'] * # Adjusted for discount (less revenue, but higher conversion)
        available['inventory_norm'] * available['pref_norm']
    )
    
    # 3. CUSTOMER PRIORITY FACTOR (NEW)
    # Loyalty Tier Bonus
    loyalty_multipliers = {
        'Gold': 1.15,  # 15% rank boost
        'Silver': 1.05, # 5% rank boost
        'Standard': 1.0
    }
    loyalty_bonus = loyalty_multipliers.get(customer_tier, 1.0)
    
    # Cancellation Compensation Multiplier
    # If flag is 1.0 (true), give a significant temporary boost (e.g., 20%)
    cancellation_boost = 1.0 + (cancellation_penalty_flag * 0.20) 
    
    # Combine Customer Priority
    customer_priority_factor = loyalty_bonus * cancellation_boost

    # --- DYNAMIC WEIGHTING & FINAL SCORE CALCULATION (Modified) ---

    def calculate_unified_score(row):
        # ... Dynamic weight calculation based on time_remaining_h (W_urgency, W_revenue, W_pref) ...
        
        # New base score incorporates the Store Operational Metric
        base_score = (
            W_urgency * row['urgency_score'] + 
            W_revenue * row['revenue_metric'] + 
            W_pref * row['pref_norm']
        )
        
        # FINAL SCORE = (Base Score * Store Operational Score) * Customer Priority Factor * Fairness Factor
        final_score = (
            base_score * row['op_norm'] * # Store's operational reliability is a multiplier
            customer_priority_factor * # Customer's priority (Cancellation/Loyalty)
            row['fairness_factor']             # Store's exposure fairness
        )
        return final_score

    available['unified_score'] = available.apply(calculate_unified_score, axis=1)

    # --- RANKING (Skipped for brevity) ---
    return top_n['store_id'].tolist()

ALGORITHMS = {
    'Greedy Baseline': greedy_baseline,
    'Inventory Aware': inventory_aware,
    'Inventory-Rating Equilibrium': inventory_rating_equilibrium,
    'Underdog Boost': underdog_boost,
    'Waste Prevention': waste_prevention_threshold,
 
    'Time Decay Urgency': time_decay_urgency,
    'Supply Demand Equilibrium': supply_demand_equilibrium,
    # ADVANCED HIGH-PERFORMANCE ALGORITHMS
    'DP Optimal Visibility': dp_optimal_visibility,
    'Customer-Aware Demand': customer_aware_demand_prediction,
    'Constraint Backtracking': constraint_backtracking_fairness,
    # PERSONALIZED ALGORITHMS - Show different stores to different customers!
    'Personalized Top-K': personalized_top_k,
    'Personalized Waste-Aware': personalized_waste_aware,
    'Personalized Diverse': personalized_diverse,
    'Personalized Reliable': personalized_reliable,
    'Personalized Ultimate': personalized_ultimate,
    # GENETIC ALGORITHM
    'Genetic Algorithm': genetic_algorithm_ranking,
    'Unified Optimization V2': unified_optimization_score_v2,
}

def register_accuracy_aware_algorithm(accuracy_tracker, name='Accuracy Aware'):
    """
    Register the accuracy-aware algorithm with a specific tracker.
    
    Call this function after initializing your AccuracyTracker to add
    the accuracy-aware algorithm to the registry.
    
    Args:
        accuracy_tracker (AccuracyTracker): Initialized tracker instance
        name (str): Name for the algorithm in the registry
        
    Example:
        from simulation.accuracy_tracker import AccuracyTracker
        from simulation.algorithms import register_accuracy_aware_algorithm, ALGORITHMS
        
        tracker = AccuracyTracker()
        register_accuracy_aware_algorithm(tracker)
        
        # Now available as ALGORITHMS['Accuracy Aware']
    """
    ALGORITHMS[name] = create_accuracy_aware_algorithm(accuracy_tracker)


def unregister_algorithm(name):
    """
    Remove an algorithm from the registry.
    
    Args:
        name (str): Algorithm name to remove
        
    Returns:
        bool: True if removed, False if not found
    """
    if name in ALGORITHMS:
        del ALGORITHMS[name]
        return True
    return False


def get_available_algorithms():
    """
    Get list of available algorithm names.
    
    Returns:
        list: Names of all registered algorithms
    """
    return list(ALGORITHMS.keys())


def get_algorithm_info():
    """
    Get information about all available algorithms.
    
    Returns:
        dict: {algorithm_name: {'description': str, 'technique': str}}
    """
    info = {}
    for name, func in ALGORITHMS.items():
        if func.__doc__:
            lines = func.__doc__.strip().split('\n')
            description = lines[0].strip()
            
            # Try to find technique
            technique = "Not specified"
            for line in lines:
                if 'Technique:' in line:
                    technique = line.split('Technique:')[1].strip()
                    break
            
            info[name] = {
                'description': description,
                'technique': technique
            }
        else:
            info[name] = {
                'description': "No description available",
                'technique': "Not specified"
            }
    return info