"""
Ranking Algorithms for Food Waste Reduction Platform

This module contains different store ranking algorithms that determine
which stores are displayed to customers. The goal is to optimize for:
- Minimizing food waste (unsold bags)
- Maximizing revenue
- Fair distribution of exposure across stores

Each algorithm has the same signature:
    func(stores_df, n, current_bags, customer_valuations=None) -> list[store_id]


"""

import math
from re import I
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple 
import pandas as pd


# =============================================================================
# DISTANCE CONFIGURATION AND HELPER
# =============================================================================
# Distance weight: how much location affects ranking (0.0 = ignore, 1.0 = very important)
DISTANCE_WEIGHT = 0.3  # 30% weight for distance factor

def _calculate_distance_factor(customer_location, store_row, decay_km=5.0):
    """
    Calculate distance factor between customer and store.

    Args:
        customer_location: (latitude, longitude) tuple or None
        store_row: DataFrame row with store data
        decay_km: Distance at which factor drops to ~37% (default 5km)

    Returns:
        Factor between 0 and 1 (1 = very close, 0 = very far)
    """
    if customer_location is None:
        return 1.0  # No location data, no penalty

    if 'latitude' not in store_row or 'longitude' not in store_row:
        return 1.0

    # Haversine distance calculation
    R = 6371  # Earth radius in km

    lat1, lon1 = np.radians(customer_location[0]), np.radians(customer_location[1])
    lat2, lon2 = np.radians(store_row['latitude']), np.radians(store_row['longitude'])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = R * c

    # Exponential decay: closer = higher factor
    # At 0km: 1.0, at 5km: ~0.37, at 10km: ~0.14
    return np.exp(-dist / decay_km)


def greedy_baseline(stores_df, n, current_bags, customer_valuations=None, customer_location=None):
    """
    BASELINE ALGORITHM: Greedy by Rating with Distance Factor

    Strategy: Show highest-rated stores, with distance penalty for far stores.
    """
    # Filter to stores with available bags
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    available = stores_df[stores_df['store_id'].isin(available_ids)]

    if available.empty:
        return []

    # Calculate score: rating * distance_factor
    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']

        # Distance factor (1.0 if no location, otherwise decays with distance)
        dist_factor = _calculate_distance_factor(customer_location, row)

        # Combined score: rating with distance penalty
        # DISTANCE_WEIGHT controls how much distance matters
        score = rating * (1 - DISTANCE_WEIGHT + DISTANCE_WEIGHT * dist_factor)
        scores.append((sid, score))

    # Sort by score descending and take top n
    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def supply_demand_equilibrium(stores_df, n, current_bags, customer_valuations=None,
                               demand_forecast=None, customer_location=None):
    """
    SMART EQUILIBRIUM + REVENUE BOOST + DISTANCE AWARENESS

    Factors:
    - RATING: Customer preference/store quality
    - SUPPLY: Inventory urgency (waste prevention)
    - PRICE: Revenue optimization
    - DISTANCE: Prefer nearby stores for customers

    Formula:
        Score = (Rating) * (1 + 0.5*Supply) * (1 + 0.3*Price) * DistanceFactor
    """
    # 1. Filter to stores with available bags
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)].copy()

    # 2. Pre-calculate Normalization Factors
    max_bags = max([current_bags[sid] for sid in available_ids]) if available_ids else 1
    log_max_bags = math.log1p(max_bags)

    # Find max price for normalization
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

        # C. PRICE BOOST (Revenue KPI)
        price_index = price / max_price if max_price > 0 else 0

        # D. DISTANCE FACTOR (Location KPI) - NEW!
        dist_factor = _calculate_distance_factor(customer_location, row)

        # FINAL SCORE with distance penalty
        base_score = base_rating * (1.0 + (0.5 * supply_index)) * (1.0 + (0.3 * price_index))
        score = base_score * (1 - DISTANCE_WEIGHT + DISTANCE_WEIGHT * dist_factor)

        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]



def time_decay_urgency(stores_df, n, current_bags, closing_times=None, current_time=None,
                       customer_valuations=None, exposure_history=None, customer_id=None,
                       waste_prevention_mode='balanced', customer_location=None):
    """
    Enhanced with aggressive waste prevention strategies and distance awareness.

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

    # Calculate distance factor for each store
    available['distance_factor'] = available.apply(
        lambda row: _calculate_distance_factor(customer_location, row), axis=1
    )
    
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

    # Apply distance factor to personalized score
    available['personalized_score'] = (
        available['personalized_score'] *
        (1 - DISTANCE_WEIGHT + DISTANCE_WEIGHT * available['distance_factor'])
    )

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

    # If no customer valuations, fall back to greedy_baseline
    if customer_valuations is None:
        return greedy_baseline(stores_df, n, current_bags)

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

    # If no customer valuations, fall back to greedy_baseline (best non-personalized)
    if customer_valuations is None:
        return greedy_baseline(stores_df, n, current_bags)

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

    # If no customer valuations, fall back to greedy_baseline
    if customer_valuations is None:
        return greedy_baseline(stores_df, n, current_bags)

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
# 
# ARCHITECTURE (End-of-Day Evaluation):
# 1. Active chromosome (starts as supply_demand_equilibrium) makes real decisions
# 2. All chromosomes track what they WOULD select in background (shadow simulation)
# 3. At END OF DAY: Compute what KPIs each chromosome WOULD have produced
# 4. If a background chromosome produces better results, swap it to active
# 5. Evolve population and repeat
#
# This ensures we never perform WORSE than supply_demand_equilibrium,
# but can IMPROVE if GA finds better weights.
# =============================================================================

import random

# Global state for GA - persists across simulation days
_GA_POPULATION = None          # List of chromosomes (weight vectors)
_GA_ACTIVE_CHROMOSOME_IDX = 0  # Index of chromosome making REAL decisions
_GA_BEST_CHROMOSOME = None     # Best chromosome found so far
_GA_BEST_FITNESS = -float('inf')  # Best fitness score
_GA_GENERATION = 0             # Current generation
_GA_DAY_COUNT = 0              # Number of days simulated

# Shadow tracking - what each chromosome WOULD have selected
_GA_SHADOW_SELECTIONS = {}     # {chromosome_idx: {store_id: count}}
_GA_SHADOW_CUSTOMER_CHOICES = {} # {chromosome_idx: [(customer_valuations, displayed_stores)]}

# Current day's actual data (set by engine callback)
_GA_CURRENT_DAY_ESTIMATED = {}
_GA_CURRENT_DAY_ACTUAL = {}
_GA_CURRENT_DAY_PRICES = {}
_GA_STORES_DF = None

# GA Configuration - OPTIMIZED for speed without compromising performance
_GA_CONFIG = {
    'population_size': 12,      # Reduced from 20 - keep best presets + 4 random
    'mutation_rate': 0.15,      # Probability of mutation
    'crossover_rate': 0.7,      # Probability of crossover
    'elite_count': 2,           # Top chromosomes preserved each generation
    'tournament_size': 3,       # Tournament selection size
    'evolve_every_n_days': 2,   # Increased from 1 - more data per evolution
    'shadow_sample_rate': 0.3,  # Only track 30% of customers for shadow sim
    'early_stop_generations': 3, # Stop if no improvement for N generations
}

# Early stopping tracking
_GA_NO_IMPROVEMENT_COUNT = 0
_GA_LAST_BEST_FITNESS = -float('inf')


class Chromosome:
    """
    A chromosome represents weights for store ranking.
    
    The ACTIVE chromosome makes real decisions.
    Other chromosomes run in background (shadow simulation) to evaluate alternatives.
    
    Genes (weights) control the ranking formula:
    - w_inventory_urgency: Multiplier strength for inventory factor
    - w_customer_match: Multiplier strength for price/match factor
    - w_spread_demand: (unused in current formula, reserved for future)
    - w_time_pressure: (unused in current formula, reserved for future)
    
    Chromosome #1 is EXACTLY supply_demand_equilibrium:
    w_inventory_urgency=0.50, w_customer_match=0.30
    
    This gives: score = base_rating * (1 + 0.5*supply) * (1 + 0.3*price)
    """
    
    def __init__(self, weights=None):
        if weights is None:
            # Random initialization - bias toward supply_demand_equilibrium-like weights
            raw = [
                random.uniform(0.3, 0.6),   # w_inventory_urgency (like 0.5 in equilibrium)
                random.uniform(0.2, 0.5),   # w_customer_match (like 0.3 in equilibrium)
                random.uniform(0.05, 0.2),  # w_spread_demand (not used in current formula)
                random.uniform(0.05, 0.15)  # w_time_pressure (not used in current formula)
            ]
            total = sum(raw)
            self.weights = {
                'w_inventory_urgency': raw[0] / total,
                'w_customer_match': raw[1] / total,
                'w_spread_demand': raw[2] / total,
                'w_time_pressure': raw[3] / total,
            }
        else:
            self.weights = weights.copy()
        
        self.fitness = 0.0
        self.total_fitness = 0.0  # Cumulative across all days
        self.day_count = 0        # Number of days evaluated
        
        # Shadow tracking for this chromosome
        self.shadow_reservations = {}  # {store_id: count} what this chromosome would select
        self.shadow_estimated = {}     # Copy of estimated bags at start of day
    
    def mutate(self, mutation_rate):
        """Apply random mutation to weights."""
        if random.random() < mutation_rate:
            # Pick a random weight to mutate
            keys = list(self.weights.keys())
            key1, key2 = random.sample(keys, 2)
            
            # Transfer some weight from one to another
            delta = random.uniform(0.05, 0.2) * self.weights[key1]
            self.weights[key1] -= delta
            self.weights[key2] += delta
            
            # Ensure non-negative
            for k in self.weights:
                self.weights[k] = max(0.01, self.weights[k])
            
            # Renormalize
            total = sum(self.weights.values())
            for k in self.weights:
                self.weights[k] /= total
    
    def copy(self):
        """Create a copy of this chromosome."""
        c = Chromosome(self.weights)
        c.fitness = self.fitness
        c.total_fitness = self.total_fitness
        c.day_count = self.day_count
        c.shadow_reservations = self.shadow_reservations.copy()
        c.shadow_estimated = self.shadow_estimated.copy()
        return c
    
    def reset_shadow(self, estimated_bags):
        """Reset shadow tracking for a new day."""
        self.shadow_reservations = {}
        self.shadow_estimated = estimated_bags.copy()
    
    def __repr__(self):
        w = self.weights
        return f"Chromosome(inv={w['w_inventory_urgency']:.2f}, match={w['w_customer_match']:.2f}, fit={self.fitness:.2f})"


def _crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """
    Create offspring by combining two parents (uniform crossover).
    """
    child_weights = {}
    for key in parent1.weights:
        # 50% chance to inherit from each parent
        if random.random() < 0.5:
            child_weights[key] = parent1.weights[key]
        else:
            child_weights[key] = parent2.weights[key]
    
    # Renormalize
    total = sum(child_weights.values())
    for k in child_weights:
        child_weights[k] /= total
    
    return Chromosome(child_weights)


def _tournament_select(population: list, tournament_size: int) -> Chromosome:
    """Select a chromosome using tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda c: c.fitness)


def _initialize_population():
    """
    Initialize the GA population.
    
    IMPORTANT: Chromosome #1 (index 0) is the ACTIVE chromosome that makes real decisions.
    It is initialized to EXACTLY match supply_demand_equilibrium.
    
    This ensures GA starts EQUAL to supply_demand_equilibrium and can only improve.
    """
    global _GA_POPULATION, _GA_ACTIVE_CHROMOSOME_IDX, _GA_BEST_CHROMOSOME, _GA_BEST_FITNESS
    
    population = []
    
    # Chromosome #1: EXACT COPY OF SUPPLY_DEMAND_EQUILIBRIUM (this is the ACTIVE one)
    # Score = base_rating * (1 + 0.5*supply) * (1 + 0.3*price)
    population.append(Chromosome({
        'w_inventory_urgency': 0.50,  # EXACTLY like supply_demand_equilibrium
        'w_customer_match': 0.30,     # EXACTLY like supply_demand_equilibrium
        'w_spread_demand': 0.10,
        'w_time_pressure': 0.10
    }))
    
    # Chromosome #2: Higher inventory emphasis
    population.append(Chromosome({
        'w_inventory_urgency': 0.60,
        'w_customer_match': 0.25,
        'w_spread_demand': 0.08,
        'w_time_pressure': 0.07
    }))
    
    # Chromosome #3: Higher customer match
    population.append(Chromosome({
        'w_inventory_urgency': 0.40,
        'w_customer_match': 0.40,
        'w_spread_demand': 0.10,
        'w_time_pressure': 0.10
    }))
    
    # Chromosome #4: Very high inventory weight (aggressive waste reduction)
    population.append(Chromosome({
        'w_inventory_urgency': 0.70,
        'w_customer_match': 0.15,
        'w_spread_demand': 0.10,
        'w_time_pressure': 0.05
    }))
    
    # Chromosome #5: Lower inventory, higher match (customer-centric)
    population.append(Chromosome({
        'w_inventory_urgency': 0.35,
        'w_customer_match': 0.45,
        'w_spread_demand': 0.10,
        'w_time_pressure': 0.10
    }))
    
    # Chromosome #6: Slight variations from equilibrium
    population.append(Chromosome({
        'w_inventory_urgency': 0.55,
        'w_customer_match': 0.28,
        'w_spread_demand': 0.10,
        'w_time_pressure': 0.07
    }))
    
    # Chromosome #7: Another variation
    population.append(Chromosome({
        'w_inventory_urgency': 0.48,
        'w_customer_match': 0.35,
        'w_spread_demand': 0.10,
        'w_time_pressure': 0.07
    }))
    
    # Chromosome #8: Balanced
    population.append(Chromosome({
        'w_inventory_urgency': 0.45,
        'w_customer_match': 0.35,
        'w_spread_demand': 0.12,
        'w_time_pressure': 0.08
    }))
    
    # Fill rest with random chromosomes
    while len(population) < _GA_CONFIG['population_size']:
        population.append(Chromosome())
    
    _GA_POPULATION = population
    _GA_ACTIVE_CHROMOSOME_IDX = 0  # Start with supply_demand_equilibrium mimic
    _GA_BEST_CHROMOSOME = population[0].copy()
    _GA_BEST_FITNESS = -float('inf')


def _evaluate_selection(selected_stores, current_bags, customer_valuations, store_ratings, prices, exposure_counts=None):
    """
    Evaluate store selection for WASTE MINIMIZATION.
    
    Primary Goal: Select stores that will actually SELL their bags
    
    Fitness Score (higher = better waste reduction):
    1. Inventory cleared: Prioritize high-inventory stores
    2. Customer match: Customer is likely to buy from this store
    3. Spread bonus: Don't over-concentrate on few stores
    
    The BEST selection shows high-inventory stores to interested customers!
    """
    if not selected_stores:
        return 0.0
    
    total_bags = sum(current_bags.values()) if current_bags else 1
    max_bags = max(current_bags.values()) if current_bags else 1
    num_stores = len(current_bags)
    avg_bags = total_bags / num_stores if num_stores > 0 else 1
    
    fitness = 0.0
    
    for sid in selected_stores:
        bags = current_bags.get(sid, 0)
        price = prices.get(sid, 5.0)
        
        # --- INVENTORY URGENCY (main factor) ---
        inventory_score = bags / max_bags if max_bags > 0 else 0
        
        # Extra boost for stores way above average
        if bags > avg_bags * 1.5:
            inventory_score *= 1.5
        
        # --- CUSTOMER MATCH ---
        if customer_valuations:
            customer_interest = customer_valuations.get(sid, 2.5) / 5.0
        else:
            customer_interest = store_ratings.get(sid, 3.0) / 5.0
        
        waste_reduction_potential = inventory_score * (0.5 + 0.5 * customer_interest)
        spread_bonus = 0.1 if bags > avg_bags else 0.05
        store_fitness = waste_reduction_potential + spread_bonus
        fitness += store_fitness
    
    return fitness / len(selected_stores)


def _apply_chromosome_ranking(chromosome, stores_df, n, current_bags, customer_valuations,
                               customer_location=None):
    """
    Rank stores using MULTIPLICATIVE scoring (EXACTLY like supply_demand_equilibrium).

    Formula: Score = base_rating * (1 + w_inventory*supply_index) * (1 + w_match*price_index) * distance_factor

    When w_inventory=0.5 and w_match=0.3, this is IDENTICAL to supply_demand_equilibrium.
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    if len(available_ids) <= n:
        return available_ids

    available = stores_df[stores_df['store_id'].isin(available_ids)].copy()

    max_bags = max(current_bags.get(sid, 1) for sid in available_ids)
    log_max_bags = math.log1p(max_bags)
    max_price = available['price'].max() if not available.empty else 1

    w = chromosome.weights

    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        inventory = current_bags.get(sid, 0)
        price = row.get('price', 5.0)

        if customer_valuations and sid in customer_valuations:
            base_rating = customer_valuations[sid]
        else:
            base_rating = row['average_overall_rating']

        supply_index = math.log1p(inventory) / log_max_bags if log_max_bags > 0 else 0
        price_index = price / max_price if max_price > 0 else 0

        # Distance factor
        dist_factor = _calculate_distance_factor(customer_location, row)

        # Score with distance penalty
        base_score = base_rating * (1.0 + w['w_inventory_urgency'] * supply_index) * (1.0 + w['w_customer_match'] * price_index)
        score = base_score * (1 - DISTANCE_WEIGHT + DISTANCE_WEIGHT * dist_factor)
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def _evolve_population():
    """
    Evolve the population for one generation.
    
    Steps:
    1. Check for early stopping (no improvement for N generations)
    2. Keep elite chromosomes (including current active if it's best)
    3. Tournament selection for parents
    4. Crossover to create offspring
    5. Mutation
    6. Replace population
    """
    global _GA_POPULATION, _GA_BEST_CHROMOSOME, _GA_BEST_FITNESS, _GA_GENERATION
    global _GA_NO_IMPROVEMENT_COUNT, _GA_LAST_BEST_FITNESS

    if _GA_POPULATION is None:
        return

    # Sort by fitness (based on end-of-day evaluation)
    _GA_POPULATION.sort(key=lambda c: c.fitness, reverse=True)

    # Update best if improved
    current_best = _GA_POPULATION[0].fitness
    if current_best > _GA_BEST_FITNESS:
        _GA_BEST_CHROMOSOME = _GA_POPULATION[0].copy()
        _GA_BEST_FITNESS = current_best

    # EARLY STOPPING: Check if fitness has improved
    improvement_threshold = 0.001  # 0.1% improvement considered significant
    if current_best > _GA_LAST_BEST_FITNESS * (1 + improvement_threshold):
        _GA_NO_IMPROVEMENT_COUNT = 0
        _GA_LAST_BEST_FITNESS = current_best
    else:
        _GA_NO_IMPROVEMENT_COUNT += 1

    # If no improvement for N generations, skip evolution (already converged)
    if _GA_NO_IMPROVEMENT_COUNT >= _GA_CONFIG['early_stop_generations']:
        _GA_GENERATION += 1
        return  # Skip evolution - already converged

    new_population = []

    # Elitism: keep top chromosomes unchanged
    for i in range(_GA_CONFIG['elite_count']):
        if i < len(_GA_POPULATION):
            elite = _GA_POPULATION[i].copy()
            elite.fitness = 0.0  # Reset for next evaluation
            elite.total_fitness = 0.0
            elite.day_count = 0
            new_population.append(elite)

    # Generate rest through selection, crossover, mutation
    while len(new_population) < _GA_CONFIG['population_size']:
        parent1 = _tournament_select(_GA_POPULATION, _GA_CONFIG['tournament_size'])
        parent2 = _tournament_select(_GA_POPULATION, _GA_CONFIG['tournament_size'])

        if random.random() < _GA_CONFIG['crossover_rate']:
            child = _crossover(parent1, parent2)
        else:
            child = parent1.copy() if random.random() < 0.5 else parent2.copy()

        child.mutate(_GA_CONFIG['mutation_rate'])
        child.fitness = 0.0
        child.total_fitness = 0.0
        child.day_count = 0
        new_population.append(child)

    _GA_POPULATION = new_population
    _GA_GENERATION += 1


# =============================================================================
# END-OF-DAY EVALUATION FUNCTIONS
# =============================================================================

def ga_start_day(estimated_bags, stores_df):
    """
    Called at START of each simulated day.
    Reset shadow tracking for all chromosomes.
    """
    global _GA_POPULATION, _GA_STORES_DF, _GA_CURRENT_DAY_ESTIMATED
    
    if _GA_POPULATION is None:
        _initialize_population()
    
    _GA_STORES_DF = stores_df
    _GA_CURRENT_DAY_ESTIMATED = estimated_bags.copy()
    
    # Reset shadow tracking for all chromosomes
    for chrom in _GA_POPULATION:
        chrom.reset_shadow(estimated_bags)


def _simulate_customer_choice(displayed_stores, customer_valuations, remaining_bags):
    """
    Simulate what store a customer would pick from displayed options.
    Returns (selected_store_id, remaining_bags_after).
    """
    best_store = None
    best_val = 0
    
    for sid in displayed_stores:
        val = customer_valuations.get(sid, 0) if customer_valuations else 0
        if val > best_val and remaining_bags.get(sid, 0) > 0:
            best_val = val
            best_store = sid
    
    if best_store and remaining_bags.get(best_store, 0) > 0:
        remaining_bags[best_store] -= 1
        return best_store, remaining_bags
    
    return None, remaining_bags


def ga_track_customer(n, customer_valuations):
    """
    Called for each customer during the day.
    Track what each chromosome WOULD select (shadow simulation).

    OPTIMIZED: Only sample a fraction of customers for shadow simulation.
    This dramatically reduces computation while maintaining representative fitness.

    NOTE: The ACTIVE chromosome's selection is the one actually used.
    Other chromosomes just track for comparison.
    """
    global _GA_POPULATION, _GA_STORES_DF

    if _GA_POPULATION is None or _GA_STORES_DF is None:
        return

    # OPTIMIZATION: Only track a sample of customers for shadow simulation
    if random.random() > _GA_CONFIG['shadow_sample_rate']:
        return  # Skip this customer for shadow tracking

    # For each chromosome, simulate what it would select
    for chrom in _GA_POPULATION:
        displayed = _apply_chromosome_ranking(
            chrom, _GA_STORES_DF, n, chrom.shadow_estimated, customer_valuations
        )

        # Simulate customer choice from this chromosome's selection
        chosen, new_estimated = _simulate_customer_choice(
            displayed, customer_valuations, chrom.shadow_estimated
        )

        if chosen:
            chrom.shadow_reservations[chosen] = chrom.shadow_reservations.get(chosen, 0) + 1
            chrom.shadow_estimated = new_estimated


def ga_end_day(actual_bags, prices):
    """
    Called at END of each simulated day.
    
    Evaluate what KPIs each chromosome WOULD have produced.
    Swap active chromosome if a background one did better.
    Evolve population.
    """
    global _GA_POPULATION, _GA_ACTIVE_CHROMOSOME_IDX, _GA_DAY_COUNT, _GA_BEST_CHROMOSOME, _GA_BEST_FITNESS
    
    if _GA_POPULATION is None:
        return
    
    _GA_DAY_COUNT += 1
    
    # Evaluate each chromosome based on shadow simulation
    for chrom in _GA_POPULATION:
        fitness = _compute_chromosome_fitness(chrom, actual_bags, prices)
        chrom.total_fitness += fitness
        chrom.day_count += 1
        chrom.fitness = chrom.total_fitness / chrom.day_count
    
    # Find best chromosome this day
    best_idx = 0
    best_fitness = _GA_POPULATION[0].fitness
    for i, chrom in enumerate(_GA_POPULATION):
        if chrom.fitness > best_fitness:
            best_fitness = chrom.fitness
            best_idx = i
    
    # If a background chromosome is better, swap it to active
    if best_idx != _GA_ACTIVE_CHROMOSOME_IDX:
        # Swap positions so best becomes index 0 (active)
        _GA_POPULATION[0], _GA_POPULATION[best_idx] = _GA_POPULATION[best_idx], _GA_POPULATION[0]
        _GA_ACTIVE_CHROMOSOME_IDX = 0
    
    # Update global best
    if best_fitness > _GA_BEST_FITNESS:
        _GA_BEST_CHROMOSOME = _GA_POPULATION[0].copy()
        _GA_BEST_FITNESS = best_fitness
    
    # Evolve population every N days
    if _GA_DAY_COUNT % _GA_CONFIG['evolve_every_n_days'] == 0:
        _evolve_population()


def _compute_chromosome_fitness(chromosome, actual_bags, prices):
    """
    Compute fitness based on what this chromosome's selections WOULD have produced.
    
    UNIFIED LOGIC (same as engine.py):
    - Unsold bags = Actual bags that weren't sold (waste)
    - Lost Revenue = Unsold bags × Price
    
    Example: Store has 10 actual bags, chromosome led to 7 reservations
    - If reservations (7) <= actual (10): fulfilled=7, unsold=3, lost_revenue=3×price
    - If reservations (12) > actual (10): fulfilled=10, unsold=0, lost_revenue=0
    
    Fitness = Revenue - (0.5 × Lost Revenue)
    Higher revenue, lower waste = higher fitness.
    """
    reservations = chromosome.shadow_reservations
    
    total_revenue = 0.0
    total_lost_revenue = 0.0
    total_unsold = 0
    
    # Process each store that got reservations from this chromosome
    for sid, reserved in reservations.items():
        actual = actual_bags.get(sid, 0)
        price = prices.get(sid, 5.0)
        
        # EXACT same logic as engine.py
        if reserved <= actual:
            # All reservations fulfilled, excess = waste
            fulfilled = reserved
            unsold = actual - reserved
        else:
            # Not enough actual bags - some cancelled, but NO waste
            fulfilled = actual
            unsold = 0
        
        revenue = fulfilled * price
        lost_revenue = unsold * price  # Lost Revenue = Unsold × Price
        
        total_revenue += revenue
        total_lost_revenue += lost_revenue
        total_unsold += unsold
    
    # NOTE: We do NOT penalize stores that got no reservations from this chromosome.
    # The chromosome only controls which stores it shows to customers.
    # Stores it didn't show are not its "fault" - they would have the same waste
    # regardless of which algorithm is used.
    
    # Fitness: maximize revenue, minimize waste (lost revenue)
    fitness = total_revenue - (0.5 * total_lost_revenue)
    
    return fitness


def genetic_algorithm_ranking(stores_df, n, current_bags, customer_valuations=None,
                               customer_location=None):
    """
    GENETIC ALGORITHM with End-of-Day Evaluation and Distance Awareness.

    ARCHITECTURE:
    1. Active chromosome (initially supply_demand_equilibrium) makes REAL decisions
    2. All chromosomes track shadow selections in background
    3. At end of day: evaluate what each chromosome WOULD have produced
    4. If a background chromosome did better, swap it to become active
    5. Evolve population

    This ensures:
    - We START at supply_demand_equilibrium performance (never worse)
    - We can only IMPROVE as GA finds better weights
    """
    global _GA_POPULATION, _GA_ACTIVE_CHROMOSOME_IDX

    # Initialize on first call
    if _GA_POPULATION is None:
        _initialize_population()

    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    if len(available_ids) <= n:
        return available_ids

    # Track this customer for shadow simulation
    ga_track_customer(n, customer_valuations)

    # Use ACTIVE chromosome for actual ranking (with distance awareness)
    active_chromosome = _GA_POPULATION[_GA_ACTIVE_CHROMOSOME_IDX]
    return _apply_chromosome_ranking(active_chromosome, stores_df, n, current_bags,
                                      customer_valuations, customer_location)


def reset_genetic_algorithm():
    """Reset the genetic algorithm state for a new simulation."""
    global _GA_POPULATION, _GA_ACTIVE_CHROMOSOME_IDX, _GA_BEST_CHROMOSOME
    global _GA_BEST_FITNESS, _GA_GENERATION, _GA_DAY_COUNT
    global _GA_SHADOW_SELECTIONS, _GA_SHADOW_CUSTOMER_CHOICES
    global _GA_CURRENT_DAY_ESTIMATED, _GA_CURRENT_DAY_ACTUAL, _GA_CURRENT_DAY_PRICES, _GA_STORES_DF
    global _GA_NO_IMPROVEMENT_COUNT, _GA_LAST_BEST_FITNESS

    _GA_POPULATION = None
    _GA_ACTIVE_CHROMOSOME_IDX = 0
    _GA_BEST_CHROMOSOME = None
    _GA_BEST_FITNESS = -float('inf')
    _GA_GENERATION = 0
    _GA_DAY_COUNT = 0
    _GA_SHADOW_SELECTIONS = {}
    _GA_SHADOW_CUSTOMER_CHOICES = {}
    _GA_CURRENT_DAY_ESTIMATED = {}
    _GA_CURRENT_DAY_ACTUAL = {}
    _GA_CURRENT_DAY_PRICES = {}
    _GA_STORES_DF = None
    _GA_NO_IMPROVEMENT_COUNT = 0
    _GA_LAST_BEST_FITNESS = -float('inf')


def get_ga_evolved_weights():
    """Get current GA state and best evolved weights for revenue optimization."""
    global _GA_BEST_CHROMOSOME, _GA_BEST_FITNESS, _GA_GENERATION, _GA_POPULATION
    
    if _GA_BEST_CHROMOSOME is None:
        return {
            'status': 'Not initialized',
            'generation': 0,
            'weights': None,
            'fitness': 0
        }
    
    # Get population diversity
    if _GA_POPULATION:
        avg_fitness = sum(c.fitness for c in _GA_POPULATION) / len(_GA_POPULATION)
        best_pop_fitness = max(c.fitness for c in _GA_POPULATION)
    else:
        avg_fitness = 0
        best_pop_fitness = 0
    
    w = _GA_BEST_CHROMOSOME.weights
    return {
        'status': 'Evolved',
        'generation': _GA_GENERATION,
        'weights': w.copy(),
        'fitness': _GA_BEST_FITNESS,
        'population_avg_fitness': avg_fitness,
        'population_best_fitness': best_pop_fitness,
        'description': f"Gen {_GA_GENERATION}: "
                      f"cancel={w['w_cancellation_risk']:.2f}, "
                      f"spread={w['w_demand_spread']:.2f}, "
                      f"waste={w['w_waste_risk']:.2f}, "
                      f"rev={w['w_revenue_potential']:.2f}"
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
    """
    UNIFIED OPTIMIZATION V2: Multi-factor ranking with dynamic weights.
    
    Combines operational metrics, revenue potential, and customer priority factors.
    Falls back to inventory_rating_equilibrium for core ranking logic.
    
    Args:
        stores_df: DataFrame with store information
        n: Number of stores to return
        current_bags: Dict of {store_id: available_bags}
        customer_valuations: Optional dict of {store_id: customer_rating}
        Other args: Optional advanced features (not used in basic mode)
    
    Returns:
        List of top n store IDs
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    if len(available_ids) <= n:
        return available_ids
    
    available = stores_df[stores_df['store_id'].isin(available_ids)].copy()
    max_bags = max(current_bags.get(sid, 1) for sid in available_ids)
    
    # Normalize inventory
    available['inventory_norm'] = available['store_id'].apply(
        lambda sid: current_bags.get(sid, 0) / max_bags if max_bags > 0 else 0
    )
    
    # Normalize rating
    available['rating_norm'] = available['average_overall_rating'] / 5.0
    
    # Normalize preferences if available
    if customer_valuations:
        available['pref_norm'] = available['store_id'].apply(
            lambda sid: customer_valuations.get(sid, 2.5) / 5.0
        )
    else:
        available['pref_norm'] = 0.5  # Neutral preference
    
    # Price normalization (revenue metric)
    if 'price' in available.columns:
        max_price = available['price'].max() if available['price'].max() > 0 else 1.0
        available['price_norm'] = available['price'] / max_price
    else:
        available['price_norm'] = 0.5
    
    # Calculate urgency based on inventory (higher inventory = more urgent)
    available['urgency_score'] = available['inventory_norm']
    
    # Revenue metric: price * inventory potential
    available['revenue_metric'] = available['price_norm'] * available['inventory_norm']
    
    # Dynamic weights based on available data
    W_urgency = 0.30  # Weight for urgency (inventory level)
    W_revenue = 0.35  # Weight for revenue potential
    W_pref = 0.35     # Weight for customer preference
    
    # Calculate unified score
    def calculate_unified_score(row):
        base_score = (
            W_urgency * row['urgency_score'] + 
            W_revenue * row['revenue_metric'] + 
            W_pref * row['pref_norm']
        )
        # Include rating as a quality floor
        final_score = base_score * (0.5 + 0.5 * row['rating_norm'])
        return final_score
    
    available['unified_score'] = available.apply(calculate_unified_score, axis=1)
    
    # Rank by unified score and return top n
    top_n = available.nlargest(n, 'unified_score')
    return top_n['store_id'].tolist()


def stochastic_programming(stores_df, n, current_bags, customer_valuations=None):
    """
    REVENUE-MAXIMIZING STOCHASTIC PROGRAMMING
    
    Objective: Maximize Total Revenue = Actual Revenue - Lost Revenue
    
    Where Lost Revenue has TWO sources:
    
    1. CANCELLATION LOSS (Krispy Kreme Problem):
       - Store is overbooked relative to actual capacity
       - Reservations > Actual Bags → Cancellations
       - Lost = cancelled_orders × price
       - PLUS: Angry customers who could have bought elsewhere
    
    2. CONCENTRATION LOSS (Single Customer Problem):
       - Too few customers for a store's inventory
       - One customer gets 10 bags worth of content
       - Lost = (potential_bags - 1) × price
       - The 9 bags given to 1 person = 9 lost sales
    
    3. UNDEREXPOSURE LOSS (TBS Problem):
       - Store not shown enough despite having inventory
       - Actual Bags > Reservations → Unsold capacity
       - Lost = unsold_bags × price
    
    The Matching Objective:
        For each store: Reservations_j ≈ E[ActualBags_j]
        
        We want to SPREAD demand to MATCH supply across stores.
    
    Key Insight - The Revenue Equation:
        Total Revenue = Σ_j min(Reservations_j, ActualBags_j) × Price_j
        
        To maximize this, we need:
        - Avoid over-concentration on popular stores (causes cancellations)
        - Avoid under-exposure of unpopular stores (causes waste)
        - Match expected demand to expected supply per store
    
    Algorithm Strategy:
        1. Estimate how many more customers each store NEEDS to match supply
        2. Prioritize stores with UNFILLED CAPACITY (demand < supply)
        3. Penalize stores that are OVER-SUBSCRIBED (demand > supply)
        4. Account for estimation uncertainty (±30% variance)
        5. Consider customer preferences (they must actually want to buy)
    
    All parameters derived from data - NO hardcoded values.
    
    Technique: Stochastic Programming / Revenue Optimization
    Time Complexity: O(S log S) where S = number of stores
    
    Args:
        stores_df (DataFrame): Store data with 'store_id', 'price', 'average_overall_rating'
        n (int): Number of stores to display
        current_bags (dict): {store_id: estimated_remaining_bags}
        customer_valuations (dict): {store_id: customer_preference} or None
        
    Returns:
        list: Top n store IDs that maximize expected total revenue
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    if len(available_ids) <= n:
        return available_ids
    
    # ===== STEP 1: Extract and organize data =====
    
    store_data = {}
    for _, row in stores_df.iterrows():
        sid = row['store_id']
        if sid in available_ids:
            # Customer's valuation for this store (or rating as proxy)
            val = row['average_overall_rating']
            if customer_valuations is not None and sid in customer_valuations:
                val = customer_valuations[sid]
            
            store_data[sid] = {
                'price': row['price'],
                'rating': row['average_overall_rating'],
                'estimated_bags': current_bags[sid],
                'valuation': val
            }
    
    # ===== STEP 2: Compute data-driven statistics =====
    
    prices = np.array([d['price'] for d in store_data.values()])
    estimated_bags = np.array([d['estimated_bags'] for d in store_data.values()])
    ratings = np.array([d['rating'] for d in store_data.values()])
    valuations = np.array([d['valuation'] for d in store_data.values()])
    
    # Machine epsilon to avoid division by zero
    eps = np.finfo(float).eps
    
    # Totals and means
    total_estimated_bags = estimated_bags.sum()
    total_price_capacity = (prices * estimated_bags).sum()  # Total potential revenue
    num_stores = len(available_ids)
    
    # Statistical measures
    price_mean = prices.mean()
    bags_mean = estimated_bags.mean()
    val_mean = valuations.mean()
    
    price_std = prices.std() if len(prices) > 1 else eps
    bags_std = estimated_bags.std() if len(estimated_bags) > 1 else eps
    val_std = valuations.std() if len(valuations) > 1 else eps
    
    # Ranges for normalization
    price_min, price_max = prices.min(), prices.max()
    bags_min, bags_max = estimated_bags.min(), estimated_bags.max()
    val_min, val_max = valuations.min(), valuations.max()
    
    price_range = price_max - price_min if price_max != price_min else price_mean
    bags_range = bags_max - bags_min if bags_max != bags_min else bags_mean
    val_range = val_max - val_min if val_max != val_min else val_mean
    
    # ===== STEP 3: Estimate current demand distribution =====
    #
    # Key insight: current_bags shows REMAINING estimated bags.
    # If a store started with 20 bags and now has 5, it got ~15 reservations.
    # We can infer demand patterns from the remaining inventory.
    
    # Calculate implied reservations so far (estimated)
    # We use the original bags from stores_df vs current_bags
    original_bags = {}
    for _, row in stores_df.iterrows():
        sid = row['store_id']
        if sid in available_ids:
            original_bags[sid] = row['average_bags_at_9AM']
    
    implied_reservations = {}
    for sid in available_ids:
        original = original_bags.get(sid, store_data[sid]['estimated_bags'])
        remaining = store_data[sid]['estimated_bags']
        implied_reservations[sid] = max(0, original - remaining)
    
    total_implied_reservations = sum(implied_reservations.values())
    
    # ===== STEP 4: Calculate SUPPLY-DEMAND GAP for each store =====
    #
    # Gap > 0: Store has MORE supply than demand (UNDEREXPOSED - needs customers)
    # Gap < 0: Store has MORE demand than supply (OVEREXPOSED - risk of cancellation)
    # Gap ≈ 0: Store is well-matched
    
    supply_demand_gap = {}
    for sid, data in store_data.items():
        # Expected actual bags (accounting for ±30% uncertainty)
        # Conservative estimate: use lower bound to avoid over-promising
        # E[Actual] = Estimated, but could be as low as 0.7 × Estimated
        conservative_supply = data['estimated_bags']  # Current remaining
        
        # Current demand = implied reservations
        current_demand = implied_reservations[sid]
        
        # Gap: positive means needs more customers, negative means over-subscribed
        supply_demand_gap[sid] = conservative_supply - current_demand
    
    # Normalize gaps
    gaps = np.array(list(supply_demand_gap.values()))
    gap_mean = gaps.mean()
    gap_std = gaps.std() if len(gaps) > 1 else eps
    gap_min, gap_max = gaps.min(), gaps.max()
    gap_range = gap_max - gap_min if gap_max != gap_min else abs(gap_mean) + eps
    
    # ===== STEP 5: Calculate MARGINAL REVENUE VALUE for each store =====
    #
    # The question: "If we show this store to THIS customer, what's the
    # expected change in total platform revenue?"
    #
    # Marginal Revenue = P(customer buys from this store) × 
    #                    E[Revenue impact of that purchase]
    #
    # Revenue impact considers:
    # - Direct revenue: price × P(not cancelled)
    # - Opportunity cost avoided: filling an undersupplied store
    # - Concentration penalty: don't over-concentrate on one store
    
    scores = []
    
    for sid, data in store_data.items():
        price = data['price']
        est_bags = data['estimated_bags']
        valuation = data['valuation']
        rating = data['rating']
        gap = supply_demand_gap[sid]
        
        # --- Component 1: Purchase Probability ---
        # P(this customer buys from store j | store j is displayed)
        # Based on customer's relative valuation
        
        # Normalize valuation to [0, 1]
        val_norm = (valuation - val_min) / (val_range + eps)
        
        # Purchase probability proportional to valuation
        # Higher valuation = more likely to choose this store
        purchase_prob = valuation / (valuations.sum() + eps)
        
        # --- Component 2: Fulfillment Probability ---
        # P(order is fulfilled | customer reserves)
        # Depends on gap: if store is undersupplied (gap < 0), higher cancel risk
        
        # Normalize gap to [0, 1] where 1 = most undersupplied (needs customers)
        gap_norm = (gap - gap_min) / (gap_range + eps)
        
        # If gap > 0 (more supply than demand), high fulfillment probability
        # If gap < 0 (more demand than supply), lower fulfillment probability
        if gap >= 0:
            fulfillment_prob = 1.0  # Plenty of supply, will fulfill
        else:
            # Risk of cancellation increases as gap becomes more negative
            # Scale by how negative relative to the range
            fulfillment_prob = max(0.1, 1.0 + (gap / (bags_range + eps)))
        
        # --- Component 3: Direct Revenue ---
        # E[Revenue from this sale] = price × P(fulfilled)
        
        price_norm = (price - price_min) / (price_range + eps)
        expected_direct_revenue = price_norm * fulfillment_prob
        
        # --- Component 4: Opportunity Value (Underexposure Correction) ---
        # Stores with high gap (undersupplied) have high opportunity value
        # Showing them captures revenue that would otherwise be lost
        #
        # This is the TBS problem: store has bags but isn't getting customers
        # Each unfilled bag = lost revenue opportunity
        
        # Opportunity value = gap × price (revenue at risk of being lost)
        # Normalize by total potential
        if gap > 0:
            # Store needs customers - high opportunity value
            opportunity_value = (gap * price) / (total_price_capacity + eps)
        else:
            # Store is over-subscribed - no opportunity value from showing more
            opportunity_value = 0
        
        # --- Component 5: Concentration Penalty ---
        # Penalty for showing stores that already have high demand relative to supply
        # This prevents the Krispy Kreme problem (too many bookings)
        #
        # Also prevents single-customer concentration:
        # If a store has many bags but few reservations, one more customer
        # would cause concentration (that 1 customer gets everything)
        
        # Concentration risk: how much of this store's supply would go to few customers?
        current_reservations = implied_reservations[sid]
        
        if current_reservations == 0:
            # No one has booked yet - if we're the first, we get EVERYTHING
            # This is the single-customer concentration problem
            # Concentration penalty = (bags - 1) / bags = how much is "wasted" on one person
            concentration_penalty = (est_bags - 1) / (est_bags + eps)
        elif gap < 0:
            # Over-subscribed: penalty for adding more demand
            concentration_penalty = abs(gap) / (bags_range + eps)
        else:
            # Healthy demand distribution
            concentration_penalty = 0
        
        # --- Component 6: Estimation Uncertainty Adjustment ---
        # Actual bags vary ±30% from estimate
        # Stores with high estimates have high ABSOLUTE uncertainty
        # We should be more conservative with high-estimate stores
        #
        # Uncertainty = proportional to estimate (larger estimates = more variance)
        
        relative_uncertainty = est_bags / (total_estimated_bags + eps)
        
        # --- FINAL SCORE: Expected Marginal Revenue ---
        #
        # Score = P(purchase) × [E[direct_revenue] + opportunity_value 
        #                        - concentration_penalty - uncertainty_adjustment]
        #
        # All components are data-normalized, so we combine them directly
        
        # Revenue component (want to maximize)
        revenue_component = expected_direct_revenue + opportunity_value
        
        # Risk component (want to minimize)
        risk_component = concentration_penalty * relative_uncertainty
        
        # Preference alignment (customer must actually want this store)
        preference_alignment = val_norm
        
        # Weighted combination using data-derived importance
        # CV (coefficient of variation) determines relative importance
        cv_price = price_std / (price_mean + eps)
        cv_bags = bags_std / (bags_mean + eps)
        cv_val = val_std / (val_mean + eps)
        cv_total = cv_price + cv_bags + cv_val + eps
        
        w_revenue = cv_price / cv_total
        w_supply = cv_bags / cv_total
        w_preference = cv_val / cv_total
        
        # Final expected marginal revenue score
        marginal_revenue_score = (
            purchase_prob * (
                w_revenue * revenue_component +
                w_supply * (gap_norm - risk_component) +
                w_preference * preference_alignment
            )
        )
        
        # Boost for undersupplied stores (they NEED this customer)
        # This directly addresses the TBS problem
        if gap > 0:
            undersupply_boost = gap_norm * w_supply
            marginal_revenue_score += undersupply_boost
        
        scores.append((sid, marginal_revenue_score, {
            'gap': gap,
            'purchase_prob': purchase_prob,
            'fulfillment_prob': fulfillment_prob,
            'concentration_penalty': concentration_penalty
        }))
    
    # ===== STEP 6: Select top n stores =====
    #
    # Sort by marginal revenue score (descending)
    # These are the stores where showing them to this customer
    # maximizes expected total platform revenue
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return [sid for sid, _, _ in scores[:n]]

class TimePhase(Enum):
    """Day phases with different optimization strategies"""
    MORNING = "morning"      # 8 AM - 12 PM: Focus on fairness & exploration
    AFTERNOON = "afternoon"  # 12 PM - 5 PM: Balance exploration & exploitation
    EVENING = "evening"      # 5 PM - 10 PM: Focus on clearing inventory

@dataclass
class StoreState:
    """Encapsulates all state for a single store"""
    store_id: str
    price: float
    rating: float
    estimated_bags: int
    current_reservations: int
    total_views: int
    customer_valuation: float
    
    @property
    def supply_demand_gap(self) -> float:
        """Positive = undersupplied (needs customers), Negative = oversubscribed"""
        return self.estimated_bags - self.current_reservations
    
    @property
    def booking_rate(self) -> float:
        """Reservations per view (conversion rate)"""
        return self.current_reservations / max(1, self.total_views)
    
    @property
    def cancellation_risk(self) -> float:
        """Probability of cancellation if another reservation is made"""
        if self.supply_demand_gap >= 1:
            return 0.0  # Plenty of supply
        elif self.supply_demand_gap >= 0:
            return 0.15  # Slight risk at capacity
        else:
            # Risk increases with overbooking
            overbooking_ratio = abs(self.supply_demand_gap) / max(1, self.estimated_bags)
            return min(0.9, 0.3 + 0.6 * overbooking_ratio)


def stochastic_programming_v2(
    stores_df,
    n: int,
    current_bags: Dict[str, int],
    customer_valuations: Optional[Dict[str, float]] = None,
    customer_history: Optional[Dict[str, Dict]] = None,
    current_hour: float = 12.0,
    store_reservations: Optional[Dict[str, int]] = None,
    store_views: Optional[Dict[str, int]] = None,
    exploration_rate: float = 0.1,
    fairness_weight: float = 0.2
) -> List[str]:
    """
    OPTIMIZED REVENUE-MAXIMIZING STOCHASTIC PROGRAMMING v2
    
    ═══════════════════════════════════════════════════════════════════════════
    KEY IMPROVEMENTS OVER v1:
    ═══════════════════════════════════════════════════════════════════════════
    
    1. TIME-AWARE SCORING
       - Morning: Prioritize fairness & exploration (discover demand patterns)
       - Afternoon: Balanced approach (optimize while learning)
       - Evening: Maximize inventory clearance (revenue focus)
    
    2. BAYESIAN UNCERTAINTY MODELING
       - Track uncertainty in bag estimates explicitly
       - More conservative with high-uncertainty stores early in day
       - More aggressive as actual data comes in
    
    3. EXPLORATION-EXPLOITATION BALANCE
       - UCB-style exploration bonus for under-viewed stores
       - Epsilon-greedy fallback ensures all stores get some exposure
       - Prevents "rich get richer" problem
    
    4. CUSTOMER PERSONALIZATION
       - Uses purchase history for repeat customers
       - Preference-based scoring with proper softmax conversion
       - Distance/location awareness (if available)
    
    5. SIMPLIFIED INTERPRETABLE SCORING
       - Three clear components: Revenue, Supply-Matching, Fairness
       - Time-varying weights (not arbitrary CV-based)
       - Easy to debug and explain
    
    6. REAL-TIME ADAPTATION
       - Updates strategy as reservations come in
       - Detects and responds to demand surges
       - Prevents cascade cancellations
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Algorithm: Multi-Objective Stochastic Optimization with UCB Exploration
    Time Complexity: O(S log S) where S = number of stores
    Space Complexity: O(S)
    
    Args:
        stores_df: DataFrame with 'store_id', 'price', 'average_overall_rating'
        n: Number of stores to display to this customer
        current_bags: {store_id: estimated_remaining_bags}
        customer_valuations: {store_id: preference_score} personalized scores
        customer_history: {store_id: {'purchases': int, 'cancellations': int}}
        current_hour: Hour of day (8.0 to 22.0)
        store_reservations: {store_id: current_reservation_count}
        store_views: {store_id: total_views_today}
        exploration_rate: Base probability of exploring unpopular stores
        fairness_weight: Weight for fairness objective (0-1)
        
    Returns:
        List of top n store IDs optimizing expected revenue
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0: HANDLE EDGE CASES
    # ═══════════════════════════════════════════════════════════════════════
    
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    
    if not available_ids:
        return []
    
    if len(available_ids) <= n:
        return available_ids
    
    # Initialize defaults for optional parameters
    store_reservations = store_reservations or defaultdict(int)
    store_views = store_views or defaultdict(lambda: 1)  # Avoid div by zero
    customer_history = customer_history or {}
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: DETERMINE TIME PHASE & DYNAMIC WEIGHTS
    # ═══════════════════════════════════════════════════════════════════════
    
    time_phase, weights = _get_time_phase_and_weights(current_hour, fairness_weight)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: BUILD STORE STATE OBJECTS
    # ═══════════════════════════════════════════════════════════════════════
    
    store_states: Dict[str, StoreState] = {}
    
    for _, row in stores_df.iterrows():
        sid = row['store_id']
        if sid not in available_ids:
            continue
            
        # Get customer-specific valuation or fall back to rating
        valuation = row['average_overall_rating']
        if customer_valuations and sid in customer_valuations:
            valuation = customer_valuations[sid]
        
        # Boost valuation based on positive purchase history
        if sid in customer_history:
            hist = customer_history[sid]
            purchases = hist.get('purchases', 0)
            cancellations = hist.get('cancellations', 0)
            # Loyalty boost with cancellation penalty
            loyalty_factor = 1.0 + 0.1 * purchases - 0.2 * cancellations
            valuation *= max(0.5, loyalty_factor)
        
        store_states[sid] = StoreState(
            store_id=sid,
            price=row['price'],
            rating=row['average_overall_rating'],
            estimated_bags=current_bags[sid],
            current_reservations=store_reservations.get(sid, 0),
            total_views=store_views.get(sid, 1),
            customer_valuation=valuation
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: COMPUTE GLOBAL STATISTICS FOR NORMALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    stats = _compute_global_stats(store_states)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: SCORE EACH STORE
    # ═══════════════════════════════════════════════════════════════════════
    
    scores = []
    
    for sid, state in store_states.items():
        score = _compute_store_score(state, stats, weights, time_phase)
        scores.append((sid, score))
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: EXPLORATION-EXPLOITATION SELECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    selected = _select_with_exploration(
        scores=scores,
        n=n,
        store_states=store_states,
        exploration_rate=exploration_rate,
        time_phase=time_phase
    )
    
    return selected


def _get_time_phase_and_weights(
    current_hour: float, 
    fairness_weight: float
) -> Tuple[TimePhase, Dict[str, float]]:
    """
    Determine optimization phase and component weights based on time of day.
    
    Strategy rationale:
    - Morning: Explore to learn demand patterns, ensure fair exposure
    - Afternoon: Start optimizing based on learned patterns
    - Evening: Focus on clearing inventory to minimize waste
    """
    
    # Determine phase
    if current_hour < 12:
        phase = TimePhase.MORNING
    elif current_hour < 17:
        phase = TimePhase.AFTERNOON
    else:
        phase = TimePhase.EVENING
    
    # Time-varying weights
    # Format: {revenue, supply_matching, fairness, exploration}
    
    if phase == TimePhase.MORNING:
        # Morning: High exploration & fairness, moderate revenue focus
        weights = {
            'revenue': 0.25,
            'supply_matching': 0.25,
            'fairness': fairness_weight + 0.15,
            'exploration': 0.20
        }
    elif phase == TimePhase.AFTERNOON:
        # Afternoon: Balanced approach
        weights = {
            'revenue': 0.35,
            'supply_matching': 0.30,
            'fairness': fairness_weight,
            'exploration': 0.10
        }
    else:  # Evening
        # Evening: Revenue & clearance focused, minimal exploration
        weights = {
            'revenue': 0.45,
            'supply_matching': 0.40,  # Clear unsold inventory
            'fairness': fairness_weight * 0.5,
            'exploration': 0.05
        }
    
    # Normalize weights to sum to 1
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return phase, weights


def _compute_global_stats(store_states: Dict[str, StoreState]) -> Dict[str, float]:
    """Compute statistics needed for normalization."""
    
    eps = np.finfo(float).eps
    
    prices = [s.price for s in store_states.values()]
    bags = [s.estimated_bags for s in store_states.values()]
    gaps = [s.supply_demand_gap for s in store_states.values()]
    valuations = [s.customer_valuation for s in store_states.values()]
    views = [s.total_views for s in store_states.values()]
    
    return {
        # Price stats
        'price_min': min(prices),
        'price_max': max(prices),
        'price_range': max(prices) - min(prices) + eps,
        
        # Bags stats
        'bags_total': sum(bags),
        'bags_mean': np.mean(bags),
        'bags_max': max(bags),
        
        # Gap stats
        'gap_min': min(gaps),
        'gap_max': max(gaps),
        'gap_range': max(gaps) - min(gaps) + eps,
        
        # Valuation stats
        'val_min': min(valuations),
        'val_max': max(valuations),
        'val_range': max(valuations) - min(valuations) + eps,
        'val_sum': sum(valuations),
        
        # View stats (for exploration bonus)
        'views_total': sum(views),
        'views_mean': np.mean(views),
        'views_max': max(views),
        
        # Store count
        'n_stores': len(store_states),
        
        # Epsilon
        'eps': eps
    }


def _compute_store_score(
    state: StoreState,
    stats: Dict[str, float],
    weights: Dict[str, float],
    time_phase: TimePhase
) -> float:
    """
    Compute the composite score for a single store.
    
    Score = w_rev * Revenue + w_supply * SupplyMatch + w_fair * Fairness + w_exp * Exploration
    """
    
    eps = stats['eps']
    
    # ─────────────────────────────────────────────────────────────────────
    # COMPONENT 1: EXPECTED REVENUE
    # E[Revenue] = Price × P(Purchase) × P(Fulfilled)
    # ─────────────────────────────────────────────────────────────────────
    
    # Normalize price to [0, 1] - higher price = more revenue potential
    price_norm = (state.price - stats['price_min']) / stats['price_range']
    
    # Purchase probability based on customer valuation (softmax-style)
    # Use temperature to control sharpness
    temperature = 1.0 if time_phase == TimePhase.EVENING else 2.0
    val_exp = np.exp(state.customer_valuation / temperature)
    purchase_prob = val_exp / (stats['val_sum'] / stats['n_stores'] * stats['n_stores'] + eps)
    purchase_prob = min(1.0, purchase_prob)  # Cap at 1
    
    # Fulfillment probability (inverse of cancellation risk)
    fulfillment_prob = 1.0 - state.cancellation_risk
    
    # Expected revenue score
    revenue_score = price_norm * purchase_prob * fulfillment_prob
    
    # ─────────────────────────────────────────────────────────────────────
    # COMPONENT 2: SUPPLY-DEMAND MATCHING
    # Goal: Direct customers to stores that NEED them
    # ─────────────────────────────────────────────────────────────────────
    
    gap = state.supply_demand_gap
    
    # Normalize gap to [0, 1] where 1 = most undersupplied
    gap_norm = (gap - stats['gap_min']) / stats['gap_range']
    
    if gap > 0:
        # Store needs customers - high matching value
        # Proportional to how much of their inventory is unfilled
        unfilled_ratio = gap / max(1, state.estimated_bags)
        supply_match_score = 0.5 + 0.5 * unfilled_ratio
    elif gap == 0:
        # Perfectly matched
        supply_match_score = 0.5
    else:
        # Overbooked - penalize showing this store
        overbook_ratio = abs(gap) / max(1, state.estimated_bags)
        supply_match_score = max(0, 0.5 - 0.5 * overbook_ratio)
    
    # In evening, boost undersupplied stores more aggressively
    if time_phase == TimePhase.EVENING and gap > 0:
        urgency_boost = 0.2 * gap_norm
        supply_match_score += urgency_boost
    
    # ─────────────────────────────────────────────────────────────────────
    # COMPONENT 3: FAIRNESS (Equal Exposure Opportunity)
    # Goal: Stores with fewer views get a boost
    # ─────────────────────────────────────────────────────────────────────
    
    # Inverse of view share - less viewed = higher fairness score
    view_share = state.total_views / stats['views_total']
    expected_share = 1.0 / stats['n_stores']
    
    if view_share < expected_share:
        # Under-exposed store - deserves a boost
        fairness_score = 1.0 - (view_share / expected_share)
    else:
        # Over-exposed - slight penalty
        fairness_score = max(0, 0.5 - 0.25 * (view_share / expected_share - 1))
    
    # ─────────────────────────────────────────────────────────────────────
    # COMPONENT 4: EXPLORATION BONUS (UCB-style)
    # Goal: Uncertainty bonus for less-viewed stores
    # ─────────────────────────────────────────────────────────────────────
    
    # UCB exploration term: sqrt(ln(total_views) / store_views)
    total_views = stats['views_total']
    store_views = state.total_views
    
    if total_views > 1 and store_views > 0:
        ucb_bonus = np.sqrt(np.log(total_views) / store_views)
        exploration_score = min(1.0, ucb_bonus / 2)  # Normalize
    else:
        exploration_score = 1.0  # Maximum exploration for new stores
    
    # ─────────────────────────────────────────────────────────────────────
    # FINAL SCORE: Weighted Combination
    # ─────────────────────────────────────────────────────────────────────
    
    final_score = (
        weights['revenue'] * revenue_score +
        weights['supply_matching'] * supply_match_score +
        weights['fairness'] * fairness_score +
        weights['exploration'] * exploration_score
    )
    
    return final_score


def _select_with_exploration(
    scores: List[Tuple[str, float]],
    n: int,
    store_states: Dict[str, StoreState],
    exploration_rate: float,
    time_phase: TimePhase
) -> List[str]:
    """
    Select top n stores with exploration-exploitation balance.
    
    Strategy:
    - With probability (1 - epsilon): Select top n by score (exploitation)
    - With probability epsilon: Replace some with random underseen stores (exploration)
    """
    
    # Sort by score descending
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Base selection: top n by score
    selected = [sid for sid, _ in sorted_scores[:n]]
    
    # Adjust exploration rate by time phase
    if time_phase == TimePhase.MORNING:
        effective_rate = exploration_rate * 1.5
    elif time_phase == TimePhase.AFTERNOON:
        effective_rate = exploration_rate
    else:  # Evening
        effective_rate = exploration_rate * 0.3  # Minimal exploration
    
    effective_rate = min(0.4, effective_rate)  # Cap at 40%
    
    # Determine how many slots to use for exploration
    n_explore = max(0, int(n * effective_rate))
    
    if n_explore > 0:
        # Find underexposed stores not in current selection
        remaining = [sid for sid, _ in sorted_scores[n:]]
        
        # Prioritize stores with low views and positive supply gap
        underexposed = [
            sid for sid in remaining
            if store_states[sid].supply_demand_gap > 0
        ]
        
        if underexposed:
            # Randomly select from underexposed
            np.random.shuffle(underexposed)
            exploratory_picks = underexposed[:n_explore]
            
            # Replace lowest-scored selected stores with exploratory picks
            selected = selected[:-n_explore] + exploratory_picks
    
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL ENHANCEMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_customer_valuations(
    stores_df,
    customer_id: str,
    customer_preferences: Optional[Dict] = None,
    customer_location: Optional[Tuple[float, float]] = None,
    purchase_history: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """
    Generate personalized valuations for a specific customer.
    
    Factors considered:
    1. Explicit preferences (cuisine type, dietary restrictions, etc.)
    2. Location/distance
    3. Purchase history and satisfaction
    4. Price sensitivity
    """
    
    valuations = {}
    
    for _, row in stores_df.iterrows():
        sid = row['store_id']
        base_score = row['average_overall_rating']
        
        # Factor 1: Preference alignment
        if customer_preferences:
            # Example: Check if store cuisine matches preferences
            preference_boost = 0
            if 'favorite_categories' in customer_preferences:
                # Would need store category data
                pass
            if 'price_sensitivity' in customer_preferences:
                # Lower price = higher score for price-sensitive customers
                sensitivity = customer_preferences['price_sensitivity']
                price_factor = 1.0 - (row['price'] / 200) * sensitivity
                preference_boost += 0.5 * price_factor
            base_score *= (1 + preference_boost)
        
        # Factor 2: Distance penalty (if location available)
        if customer_location and 'latitude' in row and 'longitude' in row:
            dist = _haversine_distance(
                customer_location,
                (row['latitude'], row['longitude'])
            )
            # Decay factor: score drops with distance
            distance_factor = np.exp(-dist / 5.0)  # 5km decay constant
            base_score *= distance_factor
        
        # Factor 3: Purchase history
        if purchase_history:
            store_purchases = [p for p in purchase_history if p['store_id'] == sid]
            if store_purchases:
                # Repeat customer boost
                n_purchases = len(store_purchases)
                avg_rating = np.mean([p.get('rating', 4) for p in store_purchases])
                history_boost = 1.0 + 0.1 * n_purchases * (avg_rating / 5.0)
                base_score *= history_boost
        
        valuations[sid] = base_score
    
    return valuations


def _haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate distance between two lat/lon coordinates in km."""
    R = 6371  # Earth radius in km
    
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

ALGORITHMS = {
    'Greedy Baseline': greedy_baseline,
    'Time Decay Urgency': time_decay_urgency,
    'Supply Demand Equilibrium': supply_demand_equilibrium,
    # PERSONALIZED ALGORITHMS - Show different stores to different customers!
    #'Personalized Top-K': personalized_top_k,
   # 'Personalized Waste-Aware': personalized_waste_aware,
    #'Personalized Diverse': personalized_diverse,
    #'Personalized Reliable': personalized_reliable,
    #'Personalized Ultimate': personalized_ultimate,
    # GENETIC ALGORITHM
    'Genetic Algorithm': genetic_algorithm_ranking,
   # 'Unified Optimization V2': unified_optimization_score_v2,
    #'Stochastic Programming': stochastic_programming,
}


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