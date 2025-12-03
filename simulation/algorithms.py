def greedy_baseline(stores_df, n, current_bags):
    """
    Select top-n stores by rating that still have bags available.
    This is the default/baseline approach used by the current system.
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    top_n = available.nlargest(n, 'average_overall_rating')
    return top_n['store_id'].tolist()


def inventory_aware(stores_df, n, current_bags):
    """
    Balanced scoring: 50% rating + 50% inventory level.
    Prioritizes stores with more unsold bags to reduce waste.
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    if not available_ids:
        return []

    available = stores_df[stores_df['store_id'].isin(available_ids)]
    max_bags = max(current_bags[sid] for sid in available_ids)

    scores = []
    for _, row in available.iterrows():
        sid = row['store_id']
        rating_score = row['average_overall_rating'] / 5.0
        inventory_score = current_bags[sid] / max_bags

        score = 0.5 * rating_score + 0.5 * inventory_score
        scores.append((sid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scores[:n]]


# placeholder for future team algorithms
def custom_algorithm_1(stores_df, n, current_bags):
    # team member implements their strategy here
    pass

def custom_algorithm_2(stores_df, n, current_bags):
    # team member implements their strategy here
    pass


ALGORITHMS = {
    'Greedy Baseline': greedy_baseline,
    'Inventory Aware': inventory_aware,
}
