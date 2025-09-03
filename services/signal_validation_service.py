"""Signal validation service for checklist and invalidation logic."""

import pandas as pd
from typing import Dict, Any, List, Tuple


class SignalValidationService:
    """Service for validating trading signals against checklist and invalidation conditions."""
    
    def __init__(self):
        """Initialize the signal validation service."""
        pass
    
    def evaluate_comparison(self, current_value: float, comparator: str, target_value: float) -> bool:
        """
        Evaluate a comparison between current value and target value.
        
        Args:
            current_value: Current value to compare
            comparator: Comparison operator ('<=', '>=', '<', '>', '==', '!=')
            target_value: Target value to compare against
            
        Returns:
            Boolean result of the comparison
        """
        if comparator == '<=':
            return current_value <= target_value
        elif comparator == '>=':
            return current_value >= target_value
        elif comparator == '<':
            return current_value < target_value
        elif comparator == '>':
            return current_value > target_value
        elif comparator == '==':
            return current_value == target_value
        elif comparator == '!=':
            return current_value != target_value
        else:
            return False
    
    def check_indicator_threshold(self, df: pd.DataFrame, condition: Dict[str, Any]) -> bool:
        """
        Check indicator threshold conditions.
        
        Args:
            df: Market data DataFrame
            condition: Condition dictionary with indicator, comparator, and value
            
        Returns:
            Boolean indicating if condition is met
        """
        indicator_name = condition['indicator']
        comparator = condition['comparator']
        threshold_value = condition['value']
        
        if indicator_name not in df.columns:
            return False
        
        current_value = df[indicator_name].iloc[-1]
        condition_met = self.evaluate_comparison(current_value, comparator, threshold_value)
        
        return condition_met
    
    def check_price_level(self, df: pd.DataFrame, condition: Dict[str, Any], llm_output: Dict[str, Any]) -> bool:
        """
        Check price level conditions.
        
        Args:
            df: Market data DataFrame
            condition: Condition dictionary with level, comparator, and value
            llm_output: LLM analysis output for technical indicator values
            
        Returns:
            Boolean indicating if condition is met
        """
        level_type = condition.get('level')
        comparator = condition.get('comparator', '>')
        level_value = condition.get('value')
        
        current_price = df['Close'].iloc[-1]
        
        if level_type == 'bollinger_middle':
            bb_middle = llm_output['technical_indicators']['BB20_2']['middle']
            condition_met = self.evaluate_comparison(current_price, comparator, bb_middle)
            return condition_met
        elif level_type == 'bollinger_upper':
            bb_upper = llm_output['technical_indicators']['BB20_2']['upper']
            condition_met = self.evaluate_comparison(current_price, comparator, bb_upper)
            return condition_met
        elif level_type == 'direct' or level_value is not None:
            # Handle direct price level comparison
            if level_type == 'direct':
                target_value = level_value
            else:
                target_value = level_value
            condition_met = self.evaluate_comparison(current_price, comparator, target_value)
            return condition_met
        else:
            return False
    
    def check_indicator_crossover(self, df: pd.DataFrame, condition: Dict[str, Any]) -> bool:
        """
        Check indicator crossover conditions.
        
        Args:
            df: Market data DataFrame
            condition: Condition dictionary with indicator and crossover condition
            
        Returns:
            Boolean indicating if condition is met (currently not implemented)
        """
        # For now, skip crossover conditions as they need more complex logic
        return False
    
    def check_sequence_condition(self, df: pd.DataFrame, condition: Dict[str, Any]) -> bool:
        """
        Check sequence conditions (e.g., consecutive candles).
        
        Args:
            df: Market data DataFrame
            condition: Condition dictionary with count, direction, and ATR multiple
            
        Returns:
            Boolean indicating if condition is met (currently not implemented)
        """
        # For now, skip sequence conditions as they need more complex logic
        return False
    
    def get_technical_indicator_conditions(self, llm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract technical indicator conditions from LLM output.
        
        Args:
            llm_output: LLM analysis output
            
        Returns:
            List of technical indicator conditions
        """
        technical_indicator_list = []
        for condition in llm_output['opening_signal']['checklist']:
            if condition.get('technical_indicator', False):
                technical_indicator_list.append(condition)
        return technical_indicator_list
    
    def indicator_checker(self, df: pd.DataFrame, llm_output: Dict[str, Any]) -> bool:
        """
        Check if all technical indicator conditions are met.
        
        Args:
            df: Market data DataFrame
            llm_output: LLM analysis output
            
        Returns:
            Boolean indicating if all conditions are met
        """
        technical_indicator_list = self.get_technical_indicator_conditions(llm_output)
        all_conditions_met = True
        conditions_met_count = 0
        total_conditions = len(technical_indicator_list)

        for condition in technical_indicator_list:
            indicator_name = condition['indicator'] 
            indicator_type = condition['type']
            condition_met = False
            
            if indicator_type == 'indicator_threshold':
                condition_met = self.check_indicator_threshold(df, condition)
            elif indicator_type == 'indicator_crossover':
                condition_met = self.check_indicator_crossover(df, condition)
            else:
                condition_met = False
            
            # Count conditions that are met
            if condition_met:
                conditions_met_count += 1
            else:
                all_conditions_met = False
        
        return all_conditions_met
    
    def invalidation_checker(self, df: pd.DataFrame, llm_output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check invalidation conditions from the LLM output.
        
        Args:
            df: Market data DataFrame
            llm_output: LLM analysis output
            
        Returns:
            Tuple of (invalidation_triggered, triggered_conditions)
        """
        invalidation_conditions = llm_output['opening_signal']['invalidation']
        invalidation_triggered = False
        triggered_conditions = []
        
        for condition in invalidation_conditions:
            condition_id = condition['id']
            condition_type = condition['type']
            condition_met = False
            
            if condition_type == 'price_breach':
                # Handle price breach conditions using shared function
                level = condition.get('level')
                comparator = condition.get('comparator', '>')
                
                if level == 'bollinger_middle':
                    # Convert to price_level format for shared function
                    price_condition = {
                        'level': 'bollinger_middle',
                        'comparator': comparator
                    }
                    condition_met = self.check_price_level(df, price_condition, llm_output)
                elif isinstance(level, (int, float)):
                    # Direct price level comparison
                    price_condition = {
                        'level': 'direct',
                        'comparator': comparator,
                        'value': level
                    }
                    condition_met = self.check_price_level(df, price_condition, llm_output)
            
            elif condition_type == 'indicator_threshold':
                # Use shared indicator threshold function
                condition_met = self.check_indicator_threshold(df, condition)
            
            elif condition_type == 'indicator_crossover':
                # Use shared crossover function
                condition_met = self.check_indicator_crossover(df, condition)
            
            elif condition_type == 'price_level':
                # Use shared price level function
                condition_met = self.check_price_level(df, condition, llm_output)
            
            elif condition_type == 'sequence':
                # Use shared sequence function
                condition_met = self.check_sequence_condition(df, condition)
            
            else:
                print(f"Unknown invalidation condition type: {condition_type}")
                condition_met = False
            
            # Check if this invalidation condition is triggered
            if condition_met:
                invalidation_triggered = True
                triggered_conditions.append(condition_id)
        
        return invalidation_triggered, triggered_conditions
    
    def validate_trading_signal(self, df: pd.DataFrame, llm_output: Dict[str, Any]) -> Tuple[bool, str, List[str], Dict[str, Any]]:
        """
        Comprehensive trading signal validation combining checklist and invalidation checks.
        
        Args:
            df: Market data DataFrame
            llm_output: LLM analysis output
            
        Returns:
            Tuple of (signal_valid, signal_status, triggered_conditions, market_values)
        """
        # Check if all checklist conditions are met
        checklist_passed = self.indicator_checker(df, llm_output)
        
        # Check if any invalidation conditions are triggered
        invalidation_triggered, triggered_conditions = self.invalidation_checker(df, llm_output)
        
        # Get current market values for the return
        current_price = df['Close'].iloc[-1]
        current_time = df['Open_time'].iloc[-1]
        current_rsi = df['RSI14'].iloc[-1] if 'RSI14' in df.columns else None
        
        market_values = {
            'current_price': current_price,
            'current_time': current_time,
            'current_rsi': current_rsi
        }
        
        if invalidation_triggered:
            return False, "invalidated", triggered_conditions, market_values
        elif checklist_passed:
            return True, "valid", [], market_values
        else:
            return False, "pending", [], market_values
