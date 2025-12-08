"""
Accuracy Tracking Module for Food Waste Reduction Platform

This module tracks the historical accuracy of bakery estimates and provides
adjusted predictions based on performance history.

Key Concepts:
- Accuracy Score: How close estimates are to actual values (0-100%)
- Adjusted Estimate: Predicted bags based on historical accuracy
- Buffer Bags: Remaining estimated bags after adjustment for redistribution

Algorithm Technique: Transform and Conquer
- Transform historical data into accuracy scores
- Use transformed data to adjust current estimates
"""

from collections import defaultdict
import numpy as np


class AccuracyTracker:
    """
    Tracks and manages bakery estimation accuracy over time.
    
    The accuracy ratio is calculated as: actual / estimated
    Values < 1.0 indicate overestimation, > 1.0 indicate underestimation
    
    Attributes:
        history (dict): {store_id: [(estimated, actual, date), ...]}
        window_size (int): Number of recent records to consider
        min_samples (int): Minimum samples before using historical accuracy
    """
    
    def __init__(self, window_size=14, min_samples=2):
        """
        Initialize the accuracy tracker.
        
        Args:
            window_size (int): Rolling window for accuracy calculation (default: 14 days)
            min_samples (int): Minimum data points before applying adjustments
        """
        self.history = defaultdict(list)
        self.window_size = window_size
        self.min_samples = min_samples
    
    def record_day(self, store_id, estimated, actual, day=None):
        """
        Record a bakery's estimated vs actual bag count for a day.
        
        Args:
            store_id: Unique store identifier
            estimated (int): Number of bags the bakery estimated
            actual (int): Actual number of bags available
            day: Optional day number for tracking
        
        Time Complexity: O(1) amortized
        """
        record = {
            'estimated': estimated,
            'actual': actual,
            'day': day,
            'ratio': actual / estimated if estimated > 0 else 1.0
        }
        self.history[store_id].append(record)
        
        # Keep only window_size most recent records (sliding window)
        if len(self.history[store_id]) > self.window_size:
            self.history[store_id] = self.history[store_id][-self.window_size:]
    
    def get_accuracy_ratio(self, store_id):
        """
        Get the accuracy ratio for a store.
        
        Ratio interpretation:
        - 0.8 means bakery typically delivers 80% of estimate
        - 1.0 means estimates are accurate
        - 1.2 means bakery typically delivers 120% of estimate
        
        Uses weighted average: recent records weighted more heavily.
        
        Args:
            store_id: Store identifier
            
        Returns:
            float: Accuracy ratio (0.0 to ~1.5, typically)
        
        Time Complexity: O(window_size)
        """
        records = self.history.get(store_id, [])
        
        if len(records) < self.min_samples:
            return 1.0  # Assume accurate if insufficient data
        
        # Weighted average: more recent = higher weight
        # This is a greedy approach - prioritize recent performance
        weights = np.linspace(0.5, 1.0, len(records))
        ratios = [r['ratio'] for r in records]
        
        weighted_ratio = np.average(ratios, weights=weights)
        # Cap ratio to reasonable bounds
        return max(0.5, min(1.5, weighted_ratio))
    
    def get_adjusted_estimate(self, store_id, estimated):
        """
        Get adjusted bag estimate based on historical accuracy.
        
        Example: 
        - Bakery estimates 10 bags
        - Historical ratio is 0.8 (delivers 80%)
        - Adjusted estimate = 10 * 0.8 = 8 bags
        
        Args:
            store_id: Store identifier
            estimated (int): Bakery's current estimate
            
        Returns:
            int: Adjusted estimate (conservative, floored)
        
        Time Complexity: O(window_size) for ratio calculation
        """
        ratio = self.get_accuracy_ratio(store_id)
        adjusted = int(estimated * ratio)
        return max(1, adjusted)  # At least 1 bag
    
    def get_buffer_bags(self, store_id, estimated):
        """
        Calculate buffer bags (difference between estimate and adjusted).
        
        These bags might be available but aren't guaranteed.
        Can be redistributed after primary allocation.
        
        Args:
            store_id: Store identifier
            estimated (int): Bakery's current estimate
            
        Returns:
            int: Number of uncertain/buffer bags
        """
        adjusted = self.get_adjusted_estimate(store_id, estimated)
        return max(0, estimated - adjusted)
    
    def get_confidence_level(self, store_id):
        """
        Get confidence level in the accuracy prediction.
        
        Returns:
            str: 'low', 'medium', or 'high'
            float: Confidence score 0-1
        """
        num_records = len(self.history.get(store_id, []))
        
        if num_records < self.min_samples:
            return 'low', 0.3
        elif num_records < self.window_size // 2:
            return 'medium', 0.6
        else:
            # Also consider variance in historical ratios
            ratios = [r['ratio'] for r in self.history[store_id]]
            variance = np.var(ratios)
            # Low variance = high confidence
            confidence = max(0.5, 1.0 - variance)
            return 'high', round(confidence, 2)
    
    def get_all_adjusted_estimates(self, estimated_bags):
        """
        Get adjusted estimates for all stores at once.
        
        Args:
            estimated_bags (dict): {store_id: estimated_bags}
            
        Returns:
            dict: {store_id: {'adjusted': int, 'buffer': int, 'ratio': float}}
        """
        result = {}
        for store_id, estimated in estimated_bags.items():
            ratio = self.get_accuracy_ratio(store_id)
            adjusted = self.get_adjusted_estimate(store_id, estimated)
            result[store_id] = {
                'estimated': estimated,
                'adjusted': adjusted,
                'buffer': estimated - adjusted,
                'ratio': round(ratio, 3)
            }
        return result
    
    def get_store_performance_summary(self, store_id):
        """
        Get comprehensive performance summary for a store.
        
        Returns:
            dict: Performance metrics
        """
        records = self.history.get(store_id, [])
        
        if not records:
            return {
                'accuracy_ratio': 1.0,
                'confidence': 'low',
                'total_overestimate': 0,
                'avg_daily_difference': 0,
                'records_count': 0,
                'trend': 'unknown'
            }
        
        ratios = [r['ratio'] for r in records]
        differences = [r['estimated'] - r['actual'] for r in records]
        
        # Calculate trend (improving or declining accuracy)
        trend = 'stable'
        if len(records) >= 5:
            recent = np.mean(ratios[-3:])
            older = np.mean(ratios[:-3])
            if recent > older + 0.05:
                trend = 'improving'
            elif recent < older - 0.05:
                trend = 'declining'
        
        conf_level, conf_score = self.get_confidence_level(store_id)
        
        return {
            'accuracy_ratio': round(self.get_accuracy_ratio(store_id), 3),
            'confidence': conf_level,
            'confidence_score': conf_score,
            'total_overestimate': sum(max(0, d) for d in differences),
            'avg_daily_difference': round(np.mean(differences), 2),
            'records_count': len(records),
            'trend': trend,
            'consistency': round(1 - np.std(ratios), 3) if len(ratios) > 1 else 1.0
        }