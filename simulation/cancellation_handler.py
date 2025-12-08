"""
Cancellation Handling Module for Food Waste Reduction Platform

Handles order cancellations with:
1. Alternative recommendations based on customer preferences
2. Priority queue for next-day orders
3. Refund and loyalty points management

Algorithm Technique: Greedy with weighted scoring for alternatives
- Greedily select best alternatives based on multiple criteria
- Prioritize fairness by boosting less popular stores
"""

from collections import defaultdict
from datetime import datetime, timedelta
import heapq


class CancellationHandler:
    """
    Manages order cancellations and customer recovery.
    
    Workflow:
    1. Order cancelled (store ran out)
    2. Offer alternatives ranked by preference + fairness
    3. If accepted ? new order
    4. If refused ? priority next day + refund + points
    
    Attributes:
        priority_queue (dict): {date: [(priority, customer_id, store_id)]}
        points_ledger (dict): {customer_id: points_balance}
        refund_ledger (dict): {customer_id: [refund_records]}
    """
    
    def __init__(self, points_per_cancellation=50, refund_amount=None):
        """
        Initialize the cancellation handler.
        
        Args:
            points_per_cancellation (int): Bonus points for inconvenience
            refund_amount: Default refund amount (None = full refund based on order)
        """
        self.priority_queue = defaultdict(list)  # {date: heap of (priority, cust, store)}
        self.points_ledger = defaultdict(int)
        self.refund_ledger = defaultdict(list)
        self.cancellation_log = []
        self.points_per_cancellation = points_per_cancellation
        self.default_refund = refund_amount
        
        # Stats tracking
        self.stats = {
            'total_cancellations': 0,
            'alternatives_offered': 0,
            'alternatives_accepted': 0,
            'refunds_issued': 0,
            'points_issued': 0,
            'priority_orders_served': 0
        }
    
    def handle_cancellation(self, customer_id, original_store_id, 
                           customer_valuations, available_stores,
                           current_bags, stores_df, order_price,
                           current_date=None):
        """
        Handle an order cancellation and generate alternatives.
        
        Algorithm: Greedy selection with multi-factor scoring
        
        Args:
            customer_id: Customer identifier
            original_store_id: Store that cancelled
            customer_valuations (dict): {store_id: preference_score}
            available_stores (list): Stores with remaining bags
            current_bags (dict): {store_id: remaining_bags}
            stores_df: DataFrame with store info
            order_price (float): Original order price for refund
            current_date: Current simulation date
            
        Returns:
            dict: Cancellation handling result
        
        Time Complexity: O(S log S) where S = number of available stores
        """
        current_date = current_date or datetime.now().date()
        
        self.stats['total_cancellations'] += 1
        
        # Log cancellation
        cancellation_record = {
            'customer_id': customer_id,
            'original_store': original_store_id,
            'order_price': order_price,
            'date': current_date,
            'timestamp': datetime.now(),
            'status': 'pending'
        }
        self.cancellation_log.append(cancellation_record)
        
        # Find alternatives using greedy weighted scoring
        alternatives = self._find_alternatives(
            customer_valuations=customer_valuations,
            available_stores=available_stores,
            current_bags=current_bags,
            stores_df=stores_df,
            exclude_store=original_store_id
        )
        
        if alternatives:
            self.stats['alternatives_offered'] += 1
            cancellation_record['alternatives_offered'] = [a[0] for a in alternatives[:5]]
            
            return {
                'status': 'alternatives_available',
                'alternatives': alternatives[:5],  # Top 5 alternatives
                'original_store': original_store_id,
                'customer_id': customer_id,
                'message': f"Found {len(alternatives)} alternative stores"
            }
        else:
            # No alternatives - immediately grant priority and refund
            result = self._process_no_alternatives(
                customer_id, original_store_id, order_price, current_date
            )
            cancellation_record['status'] = 'no_alternatives'
            cancellation_record['resolution'] = result
            return result
    
    def _find_alternatives(self, customer_valuations, available_stores,
                          current_bags, stores_df, exclude_store):
        """
        Find and rank alternative stores using greedy weighted scoring.
        
        Scoring formula (Transform and Conquer approach):
            score = w1 * preference + w2 * inventory + w3 * fairness_boost
        
        Where:
            - preference: Customer's valuation normalized (0-1)
            - inventory: Remaining bags normalized (0-1) - reduces waste
            - fairness_boost: Inverse popularity - helps less popular stores
        
        Args:
            customer_valuations: Customer preferences per store
            available_stores: Stores with bags available
            current_bags: Current inventory levels
            stores_df: Store information
            exclude_store: Original store to exclude
            
        Returns:
            list: [(store_id, score, details), ...] sorted by score desc
        
        Time Complexity: O(S log S) for sorting S stores
        """
        if not available_stores:
            return []
        
        # Filter stores with actual availability
        valid_stores = [s for s in available_stores 
                       if s != exclude_store and current_bags.get(s, 0) > 0]
        
        if not valid_stores:
            return []
        
        # Normalization factors
        max_bags = max(current_bags.get(s, 1) for s in valid_stores)
        max_valuation = 5.0  # Assuming 1-5 scale
        
        # Get ratings for fairness calculation
        ratings = stores_df.set_index('store_id')['average_overall_rating'].to_dict()
        prices = stores_df.set_index('store_id')['price'].to_dict()
        
        alternatives = []
        for store_id in valid_stores:
            # Normalize preference (0-1)
            valuation = customer_valuations.get(store_id, 2.5)
            pref_score = valuation / max_valuation
            
            # Normalize inventory - higher inventory = higher priority (waste reduction)
            inv_score = current_bags.get(store_id, 0) / max_bags
            
            # Fairness boost - lower rated stores get boost
            # This promotes less popular bakeries with higher waste
            rating = ratings.get(store_id, 3.0)
            fairness_score = 1 - (rating / 5.0)  # Lower rating = higher score
            
            # Weighted combination
            # Weights: 40% preference, 35% inventory (waste), 25% fairness
            combined_score = (
                0.40 * pref_score +
                0.35 * inv_score +
                0.25 * fairness_score
            )
            
            alternatives.append((
                store_id,
                round(combined_score, 4),
                {
                    'preference_score': round(pref_score, 3),
                    'inventory_score': round(inv_score, 3),
                    'fairness_score': round(fairness_score, 3),
                    'bags_available': current_bags.get(store_id, 0),
                    'price': prices.get(store_id, 0),
                    'rating': rating
                }
            ))
        
        # Sort by score descending (greedy: pick best first)
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives
    
    def accept_alternative(self, customer_id, accepted_store_id, current_date=None):
        """
        Process customer accepting an alternative store.
        
        Args:
            customer_id: Customer identifier
            accepted_store_id: Chosen alternative store
            current_date: Current date
            
        Returns:
            dict: Acceptance confirmation
        """
        self.stats['alternatives_accepted'] += 1
        
        # Update cancellation record
        for record in reversed(self.cancellation_log):
            if record['customer_id'] == customer_id and record['status'] == 'pending':
                record['status'] = 'alternative_accepted'
                record['accepted_store'] = accepted_store_id
                break

        return {
            'status': 'success',
            'customer_id': customer_id,
            'new_store': accepted_store_id,
            'message': 'Alternative accepted successfully'
        }
    
    def refuse_alternatives(self, customer_id, original_store_id, 
                           order_price, current_date=None):
        """
        Process customer refusing all alternatives.
        
        Grants:
        1. Full refund of original order
        2. Bonus loyalty points for inconvenience
        3. Priority access next day at original store
        
        Args:
            customer_id: Customer identifier
            original_store_id: Originally selected store
            order_price: Amount to refund
            current_date: Current date
            
        Returns:
            dict: Refund and priority details
        """
        current_date = current_date or datetime.now().date()
        
        return self._process_no_alternatives(
            customer_id, original_store_id, order_price, current_date
        )
    
    def _process_no_alternatives(self, customer_id, original_store_id,
                                 order_price, current_date):
        """
        Internal method to process refund + priority + points.
        
        Time Complexity: O(log P) for heap push where P = priority queue size
        """
        next_date = current_date + timedelta(days=1)
        
        # 1. Issue full refund
        refund_record = {
            'amount': order_price,
            'date': current_date,
            'reason': 'order_cancellation',
            'original_store': original_store_id
        }
        self.refund_ledger[customer_id].append(refund_record)
        self.stats['refunds_issued'] += 1
        
        # 2. Grant bonus points
        self.points_ledger[customer_id] += self.points_per_cancellation
        self.stats['points_issued'] += self.points_per_cancellation
        
        # 3. Add to priority queue for next day
        # Priority is negative because heapq is min-heap (lower = higher priority)
        # Priority -2 ensures cancelled customers served before regular customers
        priority = -2  
        heapq.heappush(
            self.priority_queue[next_date],
            (priority, customer_id, original_store_id)
        )
        
        # Update cancellation record
        for record in reversed(self.cancellation_log):
            if record['customer_id'] == customer_id and record['status'] == 'pending':
                record['status'] = 'refunded_with_priority'
                record['refund_amount'] = order_price
                record['points_granted'] = self.points_per_cancellation
                record['priority_date'] = next_date
                break
        
        return {
            'status': 'refund_and_priority_granted',
            'customer_id': customer_id,
            'refund_amount': order_price,
            'points_granted': self.points_per_cancellation,
            'total_points': self.points_ledger[customer_id],
            'priority_date': next_date,
            'priority_store': original_store_id,
            'message': f'Full refund + {self.points_per_cancellation} points + priority tomorrow'
        }
    
    def get_priority_customers(self, date, store_id=None):
        """
        Get customers with priority access for a given date.
        
        Args:
            date: Date to check
            store_id: Optional - filter for specific store
            
        Returns:
            list: [(customer_id, store_id, priority), ...] in priority order
        """
        if date not in self.priority_queue:
            return []
        
        # Get sorted copy (don't modify original heap)
        queue_copy = sorted(self.priority_queue[date])
        
        result = []
        for priority, cust_id, sid in queue_copy:
            if store_id is None or sid == store_id:
                result.append({
                    'customer_id': cust_id,
                    'store_id': sid,
                    'priority': abs(priority)
                })
        
        return result
    
    def serve_priority_customer(self, date, customer_id, store_id):
        """
        Mark a priority customer as served and remove from queue.
        
        Args:
            date: Service date
            customer_id: Customer served
            store_id: Store that served them
            
        Returns:
            bool: True if customer was in queue and served
        """
        if date not in self.priority_queue:
            return False
        
        # Find and remove from queue
        queue = self.priority_queue[date]
        for i, (priority, cust_id, sid) in enumerate(queue):
            if cust_id == customer_id and sid == store_id:
                queue.pop(i)
                heapq.heapify(queue)  # Restore heap property
                self.stats['priority_orders_served'] += 1
                return True
        
        return False
    
    def get_customer_points(self, customer_id):
        """Get current points balance for a customer."""
        return self.points_ledger.get(customer_id, 0)
    
    def redeem_points(self, customer_id, points_to_redeem, conversion_rate=0.1):
        """
        Redeem points for discount.
        
        Args:
            customer_id: Customer identifier
            points_to_redeem: Points to use
            conversion_rate: Points to currency (default: 10 points = 1 EGP)
            
        Returns:
            dict: Redemption result with discount amount
        """
        available = self.points_ledger.get(customer_id, 0)
        
        if points_to_redeem > available:
            return {
                'status': 'insufficient_points',
                'available': available,
                'requested': points_to_redeem
            }
        
        discount = points_to_redeem * conversion_rate
        self.points_ledger[customer_id] -= points_to_redeem
        
        return {
            'status': 'success',
            'points_redeemed': points_to_redeem,
            'discount_amount': round(discount, 2),
            'remaining_points': self.points_ledger[customer_id]
        }
    
    def get_statistics(self):
        """Get cancellation handling statistics."""
        acceptance_rate = 0
        if self.stats['alternatives_offered'] > 0:
            acceptance_rate = (self.stats['alternatives_accepted'] / 
                              self.stats['alternatives_offered'] * 100)
        
        return {
            **self.stats,
            'acceptance_rate': round(acceptance_rate, 1),
            'total_points_in_circulation': sum(self.points_ledger.values()),
            'customers_with_points': len([c for c, p in self.points_ledger.items() if p > 0])
        }