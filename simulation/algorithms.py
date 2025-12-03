def greedy_baseline(stores_df, n, current_bags):
    """
    Select top-n stores by rating that still have bags available.
    This is the default/baseline approach used by the current system.
    """
    available_ids = [sid for sid, bags in current_bags.items() if bags > 0]
    available = stores_df[stores_df['store_id'].isin(available_ids)]
    top_n = available.nlargest(n, 'average_overall_rating')
    return top_n['store_id'].tolist()


# placeholder for future team algorithms
def custom_algorithm_1(stores_df, n, current_bags):
    # team member 1 implements their strategy here
    pass

def custom_algorithm_2(stores_df, n, current_bags):
    # team member 2 implements their strategy here
    pass


ALGORITHMS = {
    'Greedy Baseline': greedy_baseline,
    # add more as team implements them
}
