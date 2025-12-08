"""
Ranking Algorithms for Food Waste Reduction Platform

This module contains different store ranking algorithms that determine
which stores are displayed to customers. The goal is to optimize for:
- Minimizing food waste (unsold bags)
- Maximizing revenue
- Fair distribution of exposure across stores

Each algorithm has the same signature:
    func(stores_df, n, current_bags) -> list[store_id]

Where:
    - stores_df: DataFrame with store information
    - n: Number of stores to return (display to customer)
    - current_bags: Dict mapping store_id -> remaining bags

Algorithm Design Techniques (allowed per course):
1. Greedy - Make locally optimal choices
2. Divide and Conquer - Break problem into subproblems
3. Transform and Conquer - Change data representation
4. Dynamic Programming - Optimal substructure + overlapping subproblems
5. Backtracking - Build solution incrementally

NOTE: Brute force is NOT allowed in this course.
"""


def greedy_baseline(stores_df, n, current_bags):
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


def inventory_aware(stores_df, n, current_bags):
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

def accuracy_aware_ranking(stores_df, n, current_bags, accuracy_tracker=None):
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


def accuracy_aware_with_buffer_redistribution(stores_df, n, current_bags, 
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
    def ranking_func(stores_df, n, current_bags):
        return accuracy_aware_ranking(stores_df, n, current_bags, accuracy_tracker)
    
    ranking_func.__doc__ = accuracy_aware_ranking.__doc__
    return ranking_func

# =============================================================================
# PLACEHOLDER FUNCTIONS FOR TEAM MEMBERS
# Each team member should implement their own algorithm here
# =============================================================================

def custom_algorithm_1(stores_df, n, current_bags):
    """
    PLACEHOLDER: Team Member 1's Algorithm

    TODO: Implement your ranking strategy here.

    Suggested approaches:
    - Round-robin for fairness
    - Distance-based (consider customer location)
    - Price-weighted scoring
    - Dynamic weights based on time of day

    Args:
        stores_df (DataFrame): Store data
        n (int): Number of stores to display
        current_bags (dict): {store_id: remaining_bags}

    Returns:
        list: Selected store IDs to display
    """
    pass

# =============================================================================
# ALGORITHM REGISTRY
# Add your algorithm here to make it available in the dashboard
# =============================================================================

ALGORITHMS = {
    'Greedy Baseline': greedy_baseline,
    'Inventory Aware': inventory_aware,
   
    # Add your algorithms here:
    # 'My Algorithm': custom_algorithm_1,
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
