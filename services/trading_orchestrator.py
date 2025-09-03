"""Main orchestrator service to coordinate all trading services."""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from .llm_service import LLMService
from .market_data_service import MarketDataService
from .technical_analysis_service import TechnicalAnalysisService
from .signal_validation_service import SignalValidationService


class TradingOrchestrator:
    """Main orchestrator that coordinates all trading services."""
    
    def __init__(self):
        """Initialize the trading orchestrator with all services."""
        self.llm_service = LLMService()
        self.market_data_service = MarketDataService()
        self.technical_analysis_service = TechnicalAnalysisService()
        self.signal_validation_service = SignalValidationService()
        
        # Create output directory if it doesn't exist
        self.output_dir = "llm_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_chart(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a trading chart image.
        
        Args:
            image_path: Path to the chart image file
            
        Returns:
            LLM analysis results
        """
        return self.llm_service.analyze_trading_chart(image_path)
    
    def save_llm_output(self, llm_output: Dict[str, Any]) -> str:
        """
        Save LLM output to JSON file with timestamp.
        
        Args:
            llm_output: LLM analysis results
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"llm_output_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(llm_output, f, indent=2, default=str)
        
        return output_path
    
    def process_trading_signal(self, llm_output: Dict[str, Any]) -> Tuple[bool, str, list, Dict[str, Any]]:
        """
        Process a trading signal by fetching market data and validating conditions.
        
        Args:
            llm_output: LLM analysis results
            
        Returns:
            Tuple of (signal_valid, signal_status, triggered_conditions, market_values)
        """
        symbol = llm_output['symbol']
        timeframe = llm_output['timeframe']
        
        # Validate timeframe
        timeframe = self.market_data_service.validate_timeframe(timeframe)
        
        # Fetch market data
        df = self.market_data_service.fetch_market_dataframe(symbol, timeframe)
        
        if df.empty:
            print("Warning: No market data available")
            return False, "no_data", [], {}
        
        # Calculate technical indicators
        df = self.technical_analysis_service.calculate_rsi14(df)
        
        # Validate trading signal
        return self.signal_validation_service.validate_trading_signal(df, llm_output)
    
    def poll_until_decision(
        self, 
        llm_output: Dict[str, Any], 
        max_cycles: Optional[int] = None
    ) -> Tuple[bool, str, list, Dict[str, Any]]:
        """
        Poll market data until a trading decision is made.
        
        Args:
            llm_output: LLM analysis results
            max_cycles: Maximum number of polling cycles (None for unlimited)
            
        Returns:
            Tuple of (signal_valid, signal_status, triggered_conditions, market_values)
        """
        symbol = llm_output['symbol']
        timeframe = llm_output['timeframe']
        
        # Validate timeframe
        timeframe = self.market_data_service.validate_timeframe(timeframe)
        
        cycles = 0
        wait_seconds = self.technical_analysis_service.calculate_timeframe_seconds(timeframe)
        
        while True:
            # Fetch current market data
            df = self.market_data_service.fetch_market_dataframe(symbol, timeframe)
            
            if df.empty:
                print("Warning: No market data available")
                return False, "no_data", [], {}
            
            # Calculate technical indicators
            df = self.technical_analysis_service.calculate_rsi14(df)
            
            # Validate trading signal
            signal_valid, signal_status, triggered_conditions, market_values = self.signal_validation_service.validate_trading_signal(df, llm_output)
            
            if signal_status != "pending":
                return signal_valid, signal_status, triggered_conditions, market_values
            
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return signal_valid, signal_status, triggered_conditions, market_values
            
            time.sleep(wait_seconds)
    
    def make_trade_decision(
        self, 
        llm_output: Dict[str, Any], 
        market_values: Dict[str, Any], 
        checklist_passed: bool, 
        invalidation_triggered: bool, 
        triggered_conditions: list
    ) -> Dict[str, Any]:
        """
        Make a final trade decision using the LLM trade gate.
        
        Args:
            llm_output: Original LLM analysis results
            market_values: Current market values
            checklist_passed: Whether checklist conditions were met
            invalidation_triggered: Whether any invalidation conditions were triggered
            triggered_conditions: List of triggered invalidation conditions
            
        Returns:
            Trade gate decision results
        """
        gate_result = self.llm_service.llm_trade_gate_decision(
            base_llm_output=llm_output,
            market_values=market_values,
            checklist_passed=checklist_passed,
            invalidation_triggered=invalidation_triggered,
            triggered_conditions=triggered_conditions,
        )
        
        # Save gate result
        gate_filename = f"llm_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        gate_path = os.path.join(self.output_dir, gate_filename)
        with open(gate_path, 'w') as f:
            json.dump(gate_result, f, indent=2, default=str)
        
        return gate_result
    
    def run_complete_analysis(self, image_path: str, max_cycles: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete trading analysis workflow.
        
        Args:
            image_path: Path to the chart image file
            max_cycles: Maximum number of polling cycles (None for unlimited)
            
        Returns:
            Complete analysis results
        """
        # Step 1: Analyze chart
        print("Step 1: Analyzing chart...")
        llm_output = self.analyze_chart(image_path)
        
        # Step 2: Save LLM output
        print("Step 2: Saving LLM output...")
        output_path = self.save_llm_output(llm_output)
        print(f"LLM output saved to: {output_path}")
        
        # Step 3: Poll until decision
        print("Step 3: Polling for trading decision...")
        signal_valid, signal_status, triggered_conditions, market_values = self.poll_until_decision(llm_output, max_cycles)
        
        print(f"Final Signal Status: {signal_status}")
        
        # Step 4: Make trade decision if signal is valid
        gate_result = None
        if signal_valid and signal_status == "valid":
            print("Step 4: Making trade decision...")
            checklist_passed = True
            invalidation_triggered_recent = len(triggered_conditions) > 0
            gate_result = self.make_trade_decision(
                llm_output=llm_output,
                market_values=market_values,
                checklist_passed=checklist_passed,
                invalidation_triggered=invalidation_triggered_recent,
                triggered_conditions=triggered_conditions,
            )
            
            if gate_result.get("should_open") is True:
                print("Trade approved by gate!")
                # Place order integration would go here
            else:
                print("Trade rejected by gate.")
        
        return {
            'llm_output': llm_output,
            'signal_valid': signal_valid,
            'signal_status': signal_status,
            'triggered_conditions': triggered_conditions,
            'market_values': market_values,
            'gate_result': gate_result
        }
