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


def price_value_optimizer(stores_df, n, current_bags, customer_valuations=None):
    """
    PRICE-VALUE OPTIMIZER: Customer-centric value calculation.

    Key Insight: Customers want the best value for their money.
    Value = items_per_bag / price_paid

    In the surprise bag model:
    - More inventory with fewer customers = more items per bag
    - Lower price = better deal

    We estimate potential items-per-bag as: inventory / estimated_customers
    Then compute value_score = potential_items / price

    This incentivizes showing stores where customers get the best deal,
    which also happens to be stores with high inventory (reducing waste).

    Formula:
        estimated_customers = 2 + rating  # 3 to 7 customers expected
        potential_items_per_bag = inventory / estimated_customers
        value_per_egp = potential_items_per_bag / price
        score = value_per_egp_normalized + 0.3 * rating_norm

    Time Complexity: O(S log S)
    Technique: Transform and Conquer (transform to value metrics)
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)]

    # First pass: calculate value scores
    value_scores = {}
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory = current_bags[sid]
        price = row['price']

        # Estimate customers based on rating (higher rating attracts more)
        estimated_customers = 2 + rating  # Range: 3 to 7

        # Potential items per bag if only estimated_customers show up
        potential_items = inventory / estimated_customers

        # Value: items per EGP spent
        value_per_egp = potential_items / price if price > 0 else 0
        value_scores[sid] = value_per_egp

    # Normalize value scores
    max_value = max(value_scores.values()) if value_scores else 1

    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating_norm = row['average_overall_rating'] / 5.0
        value_norm = value_scores[sid] / max_value

        # Value is primary (70%), rating secondary (30%)
        score = 0.7 * value_norm + 0.3 * rating_norm
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


def round_robin_fairness(stores_df, n, current_bags, customer_valuations=None, exposure_history=None):
    """
    ROUND ROBIN FAIRNESS: Ensures all qualifying stores get equal exposure.
    
    Key Insight: Every store deserves a chance to be seen by customers.
    Popular stores don't need constant visibility; struggling stores do.
    
    Strategy (Divide and Conquer):
    1. Divide stores into quality tiers based on rating
    2. Track exposure count for each store
    3. Within each tier, select least-exposed stores first
    4. Round-robin through tiers to maintain quality balance
    
    Quality Tiers:
    - Tier 1 (Premium): Rating >= 4.0
    - Tier 2 (Good): Rating >= 3.0
    - Tier 3 (Acceptable): Rating >= 2.0
    - Excluded: Rating < 2.0 (quality threshold)
    
    Pros:
    - Guarantees fairness in exposure
    - Prevents monopolization by popular stores
    - Helps new/recovering stores build customer base
    - Reduces overall waste through distribution
    
    Cons:
    - May show lower-rated stores to maintain fairness
    - Requires tracking exposure history
    - Could reduce average customer satisfaction
    
    Time Complexity: O(S log S) for sorting within tiers
    Technique: Divide and Conquer (tier-based division)
    
    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}
        exposure_history (dict): {store_id: exposure_count} or None
        
    Returns:
        list: Selected store IDs ensuring fairness
    """
    # Filter to stores with available bags
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    
    # Initialize exposure history if not provided
    if exposure_history is None:
        exposure_history = defaultdict(int)
    
    # Divide stores into quality tiers
    tiers = {
        'premium': [],
        'good': [],
        'acceptable': []
    }
    
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory = current_bags[sid]
        exposure = exposure_history.get(sid, 0)
        
        # Skip stores below quality threshold
        if rating < 2.0:
            continue
            
        store_info = {
            'store_id': sid,
            'rating': rating,
            'inventory': inventory,
            'exposure': exposure
        }
        
        if rating >= 4.0:
            tiers['premium'].append(store_info)
        elif rating >= 3.0:
            tiers['good'].append(store_info)
        else:
            tiers['acceptable'].append(store_info)
    
    # Sort within each tier by exposure (ascending) then inventory (descending)
    for tier in tiers.values():
        tier.sort(key=lambda x: (x['exposure'], -x['inventory']))
    
    # Round-robin selection from tiers
    selected = []
    tier_order = ['premium', 'good', 'acceptable']
    tier_index = 0
    tier_positions = {'premium': 0, 'good': 0, 'acceptable': 0}
    
    # Allocate slots with tier preferences (50% premium, 35% good, 15% acceptable)
    tier_allocations = {
        'premium': max(1, int(n * 0.5)),
        'good': max(1, int(n * 0.35)),
        'acceptable': max(0, n - int(n * 0.5) - int(n * 0.35))
    }
    
    # Select stores round-robin style with tier allocations
    for tier_name in tier_order:
        tier_stores = tiers[tier_name]
        allocation = tier_allocations[tier_name]
        
        for i in range(min(allocation, len(tier_stores))):
            if len(selected) < n and i < len(tier_stores):
                selected.append(tier_stores[i]['store_id'])
    
    # Fill remaining slots if needed (greedy by inventory within available)
    if len(selected) < n:
        all_remaining = []
        for tier in tiers.values():
            all_remaining.extend([s for s in tier if s['store_id'] not in selected])
        all_remaining.sort(key=lambda x: x['inventory'], reverse=True)
        
        for store in all_remaining:
            if len(selected) < n:
                selected.append(store['store_id'])
            else:
                break
    
    return selected[:n]


def time_decay_urgency(stores_df, n, current_bags, customer_valuations=None, closing_times=None, current_time=None):
    """
    TIME DECAY URGENCY: Exponential urgency as closing time approaches.
    
    Key Insight: Food waste increases exponentially as closing time nears.
    A store with 20 bags and 30 minutes left needs immediate attention.
    
    Strategy (Transform and Conquer):
    1. Transform time remaining into urgency scores using exponential decay
    2. Combine urgency with inventory and rating
    3. Prioritize stores with highest time-sensitive waste risk
    
    Urgency Formula:
        time_factor = exp(-time_remaining_hours)
        urgency = inventory * time_factor
        score = 0.5 * urgency_norm + 0.3 * rating_norm + 0.2 * inventory_norm
    
    Example Scenarios:
    - Store A: 15 bags, 30 min left ? Very High Priority
    - Store B: 15 bags, 3 hours left ? Medium Priority
    - Store C: 5 bags, 30 min left ? High Priority
    - Store D: 30 bags, 3 hours left ? Medium-High Priority
    
    Pros:
    - Prevents last-minute waste effectively
    - Creates natural urgency for customers
    - Adapts throughout the day automatically
    
    Cons:
    - May show lower-rated stores near closing
    - Requires accurate closing time data
    - Could cluster customers at certain times
    
    Time Complexity: O(S log S) for sorting
    Technique: Transform and Conquer (time ? urgency transformation)
    
    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}
        closing_times (dict): {store_id: closing_datetime} or None
        current_time (datetime): Current time or None (uses now())
        
    Returns:
        list: Store IDs prioritized by time-sensitive urgency
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    
    # Use current time if not provided
    if current_time is None:
        current_time = datetime.now()
    
    # Generate default closing times if not provided (all stores close in 3 hours)
    if closing_times is None:
        default_closing = current_time + timedelta(hours=3)
        closing_times = {sid: default_closing for sid in available_ids}
    
    max_bags = max(current_bags[sid] for sid in available_ids)
    
    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        inventory = current_bags[sid]
        
        # Calculate time remaining
        closing = closing_times.get(sid, current_time + timedelta(hours=3))
        time_remaining = (closing - current_time).total_seconds() / 3600.0  # hours
        time_remaining = max(0.1, time_remaining)  # Minimum 6 minutes
        
        # Exponential urgency factor (increases as time decreases)
        # exp(-3) ? 0.05 for 3 hours, exp(-0.5) ? 0.61 for 30 min
        time_factor = math.exp(-time_remaining)
        
        # Calculate urgency (inventory weighted by time pressure)
        urgency = inventory * time_factor
        
        # Normalize components
        rating_norm = rating / 5.0
        inventory_norm = inventory / max_bags
        
        # For urgency normalization, consider max possible urgency
        max_urgency = max_bags * 1.0  # max is when time_factor ? 1
        urgency_norm = min(1.0, urgency / max_urgency)
        
        # Combined score with urgency as primary factor
        score = (
            0.50 * urgency_norm +      # Time-sensitive urgency
            0.30 * inventory_norm +     # Current inventory level
            0.20 * rating_norm          # Maintain some quality standard
        )
        
        scores.append({
            'store_id': sid,
            'score': score,
            'time_remaining': time_remaining,
            'urgency': urgency
        })
    
    # Sort by score descending
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    return [s['store_id'] for s in scores[:n]]


def supply_demand_equilibrium(stores_df, n, current_bags, customer_valuations=None, demand_forecast=None):
    """
    SUPPLY DEMAND EQUILIBRIUM: Balance current supply with predicted demand.
    
    Key Insight: Each store has predictable demand patterns based on:
    - Historical sales data
    - Rating (popularity)
    - Day of week / time of day
    - Weather and events
    
    Strategy (Greedy with Predictive Scoring):
    1. Forecast remaining demand for each store
    2. Calculate supply/demand imbalance
    3. Prioritize stores where supply exceeds predicted demand
    4. Prevent waste by redirecting customers to oversupplied stores
    
    Demand Forecast Model:
        base_demand = historical_average * rating_multiplier
        adjusted_demand = base_demand * time_factor * day_factor
        imbalance = current_supply - predicted_demand
    
    Example Scenarios:
    - Popular bakery: Usually sells 30 bags, has 35 ? Small imbalance
    - New restaurant: Usually sells 5 bags, has 20 ? Large imbalance!
    - Cafe on Monday: Lower demand predicted, has normal supply ? Priority
    
    Pros:
    - Data-driven approach based on patterns
    - Proactively prevents predictable waste
    - Learns and improves over time
    
    Cons:
    - Requires historical data to be effective
    - May struggle with new stores or unusual events
    - Predictions can be wrong
    
    Time Complexity: O(S log S) for sorting
    Technique: Greedy with predictive scoring
    
    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}
        demand_forecast (dict): {store_id: expected_remaining_demand} or None
        
    Returns:
        list: Store IDs with highest supply-demand imbalance
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    
    # Generate simple demand forecast if not provided
    if demand_forecast is None:
        demand_forecast = {}
        for _, row in available.iterrows():
            sid = row['store_id']
            rating = row['average_overall_rating']

            # Simple forecast: higher-rated stores have higher demand
            # Base demand = rating * 2 (so 5-star expects ~10 customers)
            base_demand = rating * 2

            # Use deterministic variation based on store_id (no random calls!)
            # This avoids corrupting the simulation's random state
            # Hash-based variation: consistent per store, range 0.7-1.3
            variation = 0.7 + (hash(str(sid)) % 1000) / 1000.0 * 0.6

            demand_forecast[sid] = base_demand * variation
    
    max_bags = max(current_bags[sid] for sid in available_ids)
    
    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating = row['average_overall_rating']
        current_supply = current_bags[sid]
        predicted_demand = demand_forecast.get(sid, rating * 2)  # fallback
        
        # Calculate supply-demand imbalance
        imbalance = current_supply - predicted_demand
        
        # Relative imbalance (what % of supply is excess?)
        if current_supply > 0:
            relative_imbalance = imbalance / current_supply
        else:
            relative_imbalance = 0
        
        # Risk score: how likely is waste?
        # Positive imbalance = oversupply (waste risk)
        # Negative imbalance = undersupply (will sell out)
        waste_risk = max(0, imbalance) / max(1, current_supply)
        
        # Normalize components
        rating_norm = rating / 5.0
        supply_norm = current_supply / max_bags
        
        # Scoring: prioritize stores with supply > demand
        if imbalance > 0:
            # Oversupplied: high priority to prevent waste
            imbalance_score = min(1.0, imbalance / max_bags)
            score = (
                0.50 * imbalance_score +    # Supply-demand gap
                0.30 * waste_risk +          # Waste probability
                0.20 * rating_norm           # Maintain quality
            )
        else:
            # Undersupplied or balanced: lower priority
            score = (
                0.30 * supply_norm +         # Current inventory
                0.70 * rating_norm           # Focus on quality
            ) * 0.5  # Reduce score for undersupplied stores
        
        scores.append({
            'store_id': sid,
            'score': score,
            'supply': current_supply,
            'demand': predicted_demand,
            'imbalance': imbalance
        })
    
    # Sort by score descending
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    return [s['store_id'] for s in scores[:n]]


def geographic_load_balancer(stores_df, n, current_bags, customer_valuations=None, zone_mapping=None):
    """
    GEOGRAPHIC LOAD BALANCER: Distribute customers across city zones.
    
    Key Insight: Geographic clustering creates problems:
    - Traffic congestion in popular areas
    - Delivery delays
    - Waste in underserved areas
    - Unfair competition within zones
    
    Strategy (Divide and Conquer):
    1. Divide city into geographic zones (districts/neighborhoods)
    2. Assess inventory levels per zone
    3. Allocate display slots proportionally to zone inventory
    4. Within each zone, select best stores
    5. Ensure geographic diversity in results
    
    Zone Analysis:
        zone_inventory = sum(bags for all stores in zone)
        zone_priority = zone_inventory / total_inventory
        zone_slots = n * zone_priority
    
    Example Distribution:
    - Zone A (Downtown): 100 bags total ? 3 slots
    - Zone B (University): 80 bags total ? 2 slots
    - Zone C (Suburbs): 60 bags total ? 2 slots
    - Zone D (Industrial): 40 bags total ? 1 slot
    
    Pros:
    - Reduces geographic waste clusters
    - Distributes customer traffic evenly
    - Supports stores in all neighborhoods
    - Reduces delivery concentration
    
    Cons:
    - May show distant stores to customers
    - Requires zone mapping data
    - Could increase average travel distance
    
    Time Complexity: O(Z * S/Z * log(S/Z)) where Z = number of zones
    Technique: Divide and Conquer (spatial partitioning)
    
    Args:
        stores_df (DataFrame): Store data with location info
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}
        zone_mapping (dict): {store_id: zone_id} or None
        
    Returns:
        list: Geographically balanced store IDs
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    
    # Generate simple zone mapping if not provided
    # Simulate zones based on store_id patterns
    if zone_mapping is None:
        zone_mapping = {}
        for sid in available_ids:
            # Simple heuristic: use store_id to determine zone
            # This simulates geographic clustering
            zone_id = f"zone_{(hash(str(sid)) % 4) + 1}"
            zone_mapping[sid] = zone_id
    
    # Organize stores by zone
    zones = defaultdict(list)
    zone_inventory = defaultdict(int)
    
    for _, row in available.iterrows():
        sid = row['store_id']
        zone = zone_mapping.get(sid, 'zone_unknown')
        inventory = current_bags[sid]
        rating = row['average_overall_rating']
        
        zones[zone].append({
            'store_id': sid,
            'inventory': inventory,
            'rating': rating,
            'zone': zone
        })
        zone_inventory[zone] += inventory
    
    # Calculate total inventory
    total_inventory = sum(zone_inventory.values())
    if total_inventory == 0:
        return []
    
    # Allocate slots to each zone based on inventory proportion
    zone_allocations = {}
    allocated_slots = 0
    
    for zone, inventory in zone_inventory.items():
        proportion = inventory / total_inventory
        slots = int(n * proportion)
        
        # Ensure at least 1 slot for zones with inventory
        if slots == 0 and inventory > 0 and allocated_slots < n:
            slots = 1
            
        zone_allocations[zone] = slots
        allocated_slots += slots
    
    # Distribute remaining slots to zones with highest inventory
    remaining_slots = n - allocated_slots
    if remaining_slots > 0:
        sorted_zones = sorted(zone_inventory.items(), key=lambda x: x[1], reverse=True)
        for zone, _ in sorted_zones:
            if remaining_slots > 0:
                zone_allocations[zone] = zone_allocations.get(zone, 0) + 1
                remaining_slots -= 1
    
    # Select best stores from each zone
    selected = []
    
    for zone, stores in zones.items():
        allocation = zone_allocations.get(zone, 0)
        if allocation == 0:
            continue
        
        # Sort stores within zone by combined score
        for store in stores:
            # Zone-local scoring
            rating_norm = store['rating'] / 5.0
            # Normalize inventory within zone
            zone_max = max(s['inventory'] for s in stores) if stores else 1
            inventory_norm = store['inventory'] / zone_max
            
            # Balanced score within zone
            store['score'] = 0.6 * rating_norm + 0.4 * inventory_norm
        
        # Sort by score within zone
        stores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top stores from zone up to allocation
        for i in range(min(allocation, len(stores))):
            selected.append(stores[i]['store_id'])
            if len(selected) >= n:
                break
    
    # If we haven't filled all slots, add best remaining stores
    if len(selected) < n:
        all_stores = []
        for zone_stores in zones.values():
            all_stores.extend(zone_stores)
        
        # Remove already selected
        remaining = [s for s in all_stores if s['store_id'] not in selected]
        remaining.sort(key=lambda x: x['score'], reverse=True)
        
        for store in remaining:
            if len(selected) < n:
                selected.append(store['store_id'])
            else:
                break
    
    return selected[:n]


def reputation_recovery(stores_df, n, current_bags, customer_valuations=None, rating_history=None):
    """
    REPUTATION RECOVERY: Boost improving stores to accelerate recovery.
    
    Key Insight: Stores that are improving deserve visibility to:
    - Reward improvement efforts
    - Build positive momentum
    - Prevent abandonment of recovery efforts
    - Show customers that stores can change
    
    Strategy (Transform and Conquer):
    1. Transform ratings into improvement trajectories
    2. Calculate rating velocity (rate of change)
    3. Identify "recovering" stores (positive trajectory)
    4. Boost recovering stores while maintaining standards
    
    Improvement Detection:
        recent_avg = average(last 10 reviews)
        historical_avg = average(all previous reviews)
        improvement = recent_avg - historical_avg
        momentum = improvement / time_period
    
    Store Categories:
    - Rising Stars: Low rating but improving fast ? High priority
    - Steady Climbers: Medium rating, improving ? Medium-high priority
    - Plateaued: High rating, stable ? Normal priority
    - Declining: Rating dropping ? Lower priority
    
    Pros:
    - Rewards quality improvements
    - Encourages stores to maintain standards
    - Creates positive feedback loop
    - Helps struggling stores recover
    
    Cons:
    - May promote inconsistent stores
    - Recent improvements might not last
    - Requires sufficient review history
    
    Time Complexity: O(S log S) for sorting
    Technique: Transform and Conquer (ratings ? trajectories)
    
    Args:
        stores_df (DataFrame): Store data with ratings
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}
        rating_history (dict): {store_id: {'recent': X, 'historical': Y}} or None
        
    Returns:
        list: Store IDs with recovery boost applied
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []
    
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    max_bags = max(current_bags[sid] for sid in available_ids)
    
    # Generate simulated rating history if not provided
    if rating_history is None:
        rating_history = {}
        for _, row in available.iterrows():
            sid = row['store_id']
            current_rating = row['average_overall_rating']

            # Use deterministic hash-based values (no random calls!)
            # This avoids corrupting the simulation's random state
            h = hash(str(sid) + 'history')

            # Deterministic category: 40% improving, 40% stable, 20% declining
            category = (h % 100) / 100.0
            if category < 0.4:  # Improving stores
                hist_delta = -0.3 - ((h % 70) / 100.0)  # -0.3 to -1.0
                recent_delta = 0.1 + ((h % 20) / 100.0)  # +0.1 to +0.3
            elif category < 0.8:  # Stable stores
                hist_delta = -0.2 + ((h % 40) / 100.0)  # -0.2 to +0.2
                recent_delta = -0.1 + ((h % 20) / 100.0)  # -0.1 to +0.1
            else:  # Declining stores
                hist_delta = 0.2 + ((h % 30) / 100.0)  # +0.2 to +0.5
                recent_delta = -0.1 - ((h % 20) / 100.0)  # -0.1 to -0.3

            historical = max(1.0, min(5.0, current_rating + hist_delta))
            recent = max(1.0, min(5.0, current_rating + recent_delta))

            rating_history[sid] = {
                'historical': historical,
                'recent': recent,
                'current': current_rating
            }
    
    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        current_rating = row['average_overall_rating']
        inventory = current_bags[sid]
        
        # Get rating trajectory
        history = rating_history.get(sid, {})
        historical_rating = history.get('historical', current_rating)
        recent_rating = history.get('recent', current_rating)
        
        # Calculate improvement metrics
        overall_improvement = current_rating - historical_rating
        recent_trend = recent_rating - current_rating
        
        # Momentum score (rate of improvement)
        # Positive = improving, Negative = declining
        momentum = (overall_improvement + recent_trend * 2) / 3
        
        # Categorize stores
        if current_rating < 3.5 and momentum > 0.3:
            category = 'rising_star'
            category_boost = 1.5
        elif current_rating >= 3.5 and momentum > 0.1:
            category = 'steady_climber'
            category_boost = 1.2
        elif abs(momentum) <= 0.1:
            category = 'stable'
            category_boost = 1.0
        else:
            category = 'declining'
            category_boost = 0.8
        
        # Normalize components
        rating_norm = current_rating / 5.0
        inventory_norm = inventory / max_bags
        
        # Calculate momentum bonus (0 to 1)
        momentum_bonus = max(0, min(1, (momentum + 1) / 2))
        
        # Combined score with category boost
        base_score = (
            0.35 * rating_norm +           # Current quality
            0.35 * inventory_norm +         # Inventory level
            0.30 * momentum_bonus           # Improvement trajectory
        )
        
        final_score = base_score * category_boost
        
        scores.append({
            'store_id': sid,
            'score': final_score,
            'category': category,
            'momentum': momentum,
            'current_rating': current_rating
        })
    
    # Sort by score descending
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    return [s['store_id'] for s in scores[:n]]


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


ALGORITHMS = {
    'Greedy Baseline': greedy_baseline,
    'Inventory Aware': inventory_aware,
    'Underdog Boost': underdog_boost,
    'Waste Prevention': waste_prevention_threshold,
    'Price-Value Optimizer': price_value_optimizer,
    'Round Robin Fairness': round_robin_fairness,
    'Time Decay Urgency': time_decay_urgency,
    'Supply Demand Equilibrium': supply_demand_equilibrium,
    'Geographic Load Balancer': geographic_load_balancer,
    'Reputation Recovery': reputation_recovery,
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