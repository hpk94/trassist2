"""Core trading analysis service for chart analysis and signal validation."""

import os
import json
import base64
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from .indicator_service import (
    calculate_rsi14,
    calculate_macd12_26_9,
    calculate_stoch14_3_3,
    calculate_bb20_2,
    calculate_atr14,
)
from .notification_service import notify_valid_trade, notify_invalidated_trade
from prompt import OPENAI_VISION_PROMPT, TRADE_GATE_PROMPT

# Load environment variables
load_dotenv()

class TradingService:
    """Service for handling trading analysis and signal validation."""
    
    def __init__(self):
        self.client = OpenAI()
        self.output_dir = "llm_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize MEXC clients when needed
        self._spot_client = None
        self._public_spot_client = None
        self._futures_client = None
    
    @property
    def spot_client(self):
        """Lazy initialization of authenticated MEXC spot client."""
        if self._spot_client is None:
            from pymexc import spot
            self._spot_client = spot.HTTP(
                api_key=os.getenv("MEXC_API_KEY"), 
                api_secret=os.getenv("MEXC_API_SECRET")
            )
        return self._spot_client
    
    @property
    def public_spot_client(self):
        """Lazy initialization of public MEXC spot client."""
        if self._public_spot_client is None:
            from pymexc import spot
            self._public_spot_client = spot.HTTP()
        return self._public_spot_client
    
    @property
    def futures_client(self):
        """Lazy initialization of MEXC futures client."""
        if self._futures_client is None:
            from pymexc import futures
            self._futures_client = futures.HTTP(
                api_key=os.getenv("MEXC_API_KEY"), 
                api_secret=os.getenv("MEXC_API_SECRET")
            )
        return self._futures_client

    def analyze_trading_chart(self, image_path: str = None, image_data: bytes = None) -> Dict[str, Any]:
        """Analyze a trading chart using OpenAI Vision API.
        
        Args:
            image_path: Path to image file (optional if image_data provided)
            image_data: Raw image bytes (optional if image_path provided)
            
        Returns:
            Dictionary containing the LLM analysis
        """
        if image_data:
            # Use provided image data
            image_b64 = base64.b64encode(image_data).decode("utf-8")
        elif image_path:
            # Read from file path
            with open(image_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            raise ValueError("Either image_path or image_data must be provided")
        
        response = self.client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": OPENAI_VISION_PROMPT},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI API returned None content")
        
        return json.loads(content)

    def save_analysis_result(self, analysis: Dict[str, Any], prefix: str = "llm_output") -> str:
        """Save analysis result to JSON file with timestamp.
        
        Args:
            analysis: Analysis result dictionary
            prefix: File prefix for naming
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return filepath

    def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        """Fetch raw klines data from MEXC API.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Chart timeframe (e.g., 1m, 5m, 1h)
            limit: Number of klines to fetch
            
        Returns:
            List of klines data
        """
        try:
            klines = self.public_spot_client.klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            return klines or []
        except Exception as e:
            print(f"Error retrieving klines for {symbol} on {timeframe}: {e}")
            return []

    def fetch_market_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch market data and return as processed DataFrame.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            
        Returns:
            DataFrame with OHLCV data and timestamps
        """
        klines = self.fetch_market_data(symbol, timeframe)
        
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines)
        df.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume']
        
        # Convert timestamps to readable datetime
        df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
        df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')
        
        # Convert price columns to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])
        
        return df

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators calculated
        """
        if df.empty:
            return df
        
        df = calculate_rsi14(df)
        df = calculate_macd12_26_9(df)
        df = calculate_stoch14_3_3(df)
        df = calculate_bb20_2(df)
        df = calculate_atr14(df)
        
        return df

    def validate_timeframe(self, timeframe: str) -> str:
        """Validate timeframe against MEXC supported intervals.
        
        Args:
            timeframe: Input timeframe
            
        Returns:
            Valid timeframe (defaults to 1m if invalid)
        """
        valid_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1W', '1M']
        if timeframe not in valid_intervals:
            print(f"Warning: {timeframe} is not a valid MEXC interval. Using 1m as default.")
            return '1m'
        return timeframe

    def validate_trading_signal(self, df: pd.DataFrame, llm_output: Dict[str, Any]) -> Tuple[bool, str, List[str], Dict[str, Any]]:
        """Comprehensive trading signal validation.
        
        Args:
            df: DataFrame with market data and indicators
            llm_output: LLM analysis output
            
        Returns:
            Tuple of (signal_valid, signal_status, triggered_conditions, market_values)
        """
        # Check if all checklist conditions are met
        checklist_passed = self._check_indicator_conditions(df, llm_output)
        
        # Check if any invalidation conditions are triggered
        invalidation_triggered, triggered_conditions = self._check_invalidation_conditions(df, llm_output)
        
        # Get current market values
        market_values = self._get_current_market_values(df)
        
        if invalidation_triggered:
            # Send invalidation notification
            try:
                trade_data = {
                    "symbol": llm_output.get("symbol", "Unknown"),
                    "current_price": market_values.get("current_price", 0),
                    "triggered_conditions": triggered_conditions,
                    "current_rsi": market_values.get("current_rsi", 0)
                }
                notify_invalidated_trade(trade_data)
            except Exception as e:
                print(f"Invalidation notification failed: {str(e)}")
            
            return False, "invalidated", triggered_conditions, market_values
        elif checklist_passed:
            return True, "valid", [], market_values
        else:
            return False, "pending", [], market_values

    def _check_indicator_conditions(self, df: pd.DataFrame, llm_output: Dict[str, Any]) -> bool:
        """Check if all indicator conditions from the checklist are met."""
        if df.empty or 'opening_signal' not in llm_output:
            return False
        
        checklist = llm_output['opening_signal'].get('checklist', [])
        technical_indicators = [item for item in checklist if item.get('technical_indicator', False)]
        
        for condition in technical_indicators:
            if not self._evaluate_condition(df, condition, llm_output):
                return False
        
        return True

    def _check_invalidation_conditions(self, df: pd.DataFrame, llm_output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check invalidation conditions from the LLM output."""
        if df.empty or 'opening_signal' not in llm_output:
            return False, []
        
        invalidation_conditions = llm_output['opening_signal'].get('invalidation', [])
        triggered_conditions = []
        
        for condition in invalidation_conditions:
            if self._evaluate_condition(df, condition, llm_output):
                triggered_conditions.append(condition.get('id', 'unknown'))
        
        return len(triggered_conditions) > 0, triggered_conditions

    def _evaluate_condition(self, df: pd.DataFrame, condition: Dict[str, Any], llm_output: Dict[str, Any]) -> bool:
        """Evaluate a single condition against the market data."""
        condition_type = condition.get('type')
        
        if condition_type == 'indicator_threshold':
            return self._check_indicator_threshold(df, condition)
        elif condition_type == 'price_level':
            return self._check_price_level(df, condition, llm_output)
        elif condition_type == 'price_breach':
            return self._check_price_level(df, condition, llm_output)
        elif condition_type == 'indicator_condition':
            return self._check_indicator_condition(df, condition)
        elif condition_type == 'indicator_crossover':
            return self._check_indicator_crossover(df, condition)
        else:
            return False

    def _check_indicator_threshold(self, df: pd.DataFrame, condition: Dict[str, Any]) -> bool:
        """Check indicator threshold conditions."""
        indicator_name = condition.get('indicator')
        comparator = condition.get('comparator')
        threshold_value = condition.get('value')
        
        if df.empty or indicator_name not in df.columns:
            return False
        
        current_value = df[indicator_name].iloc[-1]
        if pd.isna(current_value):
            return False
        
        return self._evaluate_comparison(current_value, comparator, threshold_value)

    def _check_price_level(self, df: pd.DataFrame, condition: Dict[str, Any], llm_output: Dict[str, Any]) -> bool:
        """Check price level conditions."""
        if df.empty or 'Close' not in df.columns:
            return False
        
        current_price = df['Close'].iloc[-1]
        if pd.isna(current_price):
            return False
        
        level_type = condition.get('level')
        comparator = condition.get('comparator', '>')
        level_value = condition.get('value')
        
        if level_type == 'bollinger_middle':
            bb_middle = llm_output.get('technical_indicators', {}).get('BB20_2', {}).get('middle')
            if bb_middle:
                return self._evaluate_comparison(current_price, comparator, bb_middle)
        elif level_type == 'bollinger_upper':
            bb_upper = llm_output.get('technical_indicators', {}).get('BB20_2', {}).get('upper')
            if bb_upper:
                return self._evaluate_comparison(current_price, comparator, bb_upper)
        elif level_type == 'bollinger_lower':
            bb_lower = llm_output.get('technical_indicators', {}).get('BB20_2', {}).get('lower')
            if bb_lower:
                return self._evaluate_comparison(current_price, comparator, bb_lower)
        elif level_value is not None:
            return self._evaluate_comparison(current_price, comparator, level_value)
        
        return False

    def _check_indicator_condition(self, df: pd.DataFrame, condition: Dict[str, Any]) -> bool:
        """Check indicator-specific conditions (MACD, Stochastic, etc.)."""
        indicator_name = condition.get('indicator')
        condition_text = condition.get('condition', '').lower()
        
        if indicator_name == 'MACD12_26_9':
            return self._check_macd_condition(df, condition_text)
        elif indicator_name == 'STOCH14_3_3':
            return self._check_stoch_condition(df, condition_text)
        
        return False

    def _check_macd_condition(self, df: pd.DataFrame, condition_text: str) -> bool:
        """Check MACD-specific conditions."""
        if df.empty or len(df) < 2:
            return False
        
        required_cols = ['MACD_Line', 'MACD_Signal', 'MACD_Histogram']
        if not all(col in df.columns for col in required_cols):
            return False
        
        macd_line = df['MACD_Line'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        macd_histogram = df['MACD_Histogram'].iloc[-1]
        
        if any(pd.isna(val) for val in [macd_line, macd_signal, macd_histogram]):
            return False
        
        if 'histogram increasing' in condition_text and len(df) >= 2:
            prev_histogram = df['MACD_Histogram'].iloc[-2]
            return not pd.isna(prev_histogram) and macd_histogram > prev_histogram
        elif 'histogram decreasing' in condition_text and len(df) >= 2:
            prev_histogram = df['MACD_Histogram'].iloc[-2]
            return not pd.isna(prev_histogram) and macd_histogram < prev_histogram
        elif 'macd_line < signal_line' in condition_text:
            return macd_line < macd_signal
        elif 'macd_line > signal_line' in condition_text:
            return macd_line > macd_signal
        
        return macd_histogram > 0

    def _check_stoch_condition(self, df: pd.DataFrame, condition_text: str) -> bool:
        """Check Stochastic-specific conditions."""
        if df.empty or not all(col in df.columns for col in ['STOCH_K', 'STOCH_D']):
            return False
        
        k_percent = df['STOCH_K'].iloc[-1]
        d_percent = df['STOCH_D'].iloc[-1]
        
        if any(pd.isna(val) for val in [k_percent, d_percent]):
            return False
        
        if 'k_percent < d_percent' in condition_text:
            return k_percent < d_percent
        elif 'k_percent > d_percent' in condition_text:
            return k_percent > d_percent
        
        return k_percent > d_percent

    def _check_indicator_crossover(self, df: pd.DataFrame, condition: Dict[str, Any]) -> bool:
        """Check indicator crossover conditions."""
        # For now, skip crossover conditions as they need more complex logic
        return False

    def _evaluate_comparison(self, current_value: float, comparator: str, target_value: float) -> bool:
        """Evaluate a comparison between current value and target value."""
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

    def _get_current_market_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get current market values from the DataFrame."""
        if df.empty:
            return {
                'current_price': 0,
                'current_time': None,
                'current_rsi': None
            }
        
        return {
            'current_price': df['Close'].iloc[-1] if 'Close' in df.columns else 0,
            'current_time': df['Open_time'].iloc[-1] if 'Open_time' in df.columns else None,
            'current_rsi': df['RSI14'].iloc[-1] if 'RSI14' in df.columns else None
        }

    def llm_trade_gate_decision(self, base_llm_output: Dict[str, Any], market_values: Dict[str, Any], 
                               checklist_passed: bool, invalidation_triggered: bool, 
                               triggered_conditions: List[str]) -> Dict[str, Any]:
        """Make final trade decision using LLM gate."""
        gate_context = {
            "llm_snapshot": {
                "symbol": base_llm_output.get("symbol"),
                "timeframe": base_llm_output.get("timeframe"),
                "opening_signal": base_llm_output.get("opening_signal"),
                "risk_management": base_llm_output.get("risk_management"),
            },
            "market_values": {
                "current_price": float(market_values.get("current_price", 0) or 0),
                "current_rsi": float(market_values.get("current_rsi", 0) or 0),
                "current_time": str(market_values.get("current_time")),
            },
            "program_checks": {
                "checklist_passed": bool(checklist_passed),
                "invalidation_triggered": bool(invalidation_triggered),
                "triggered_conditions": triggered_conditions,
            },
        }

        response = self.client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": TRADE_GATE_PROMPT},
                {"role": "user", "content": json.dumps(gate_context)},
            ],
            response_format={"type": "json_object"},
        )

        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            return {
                "should_open": False,
                "direction": base_llm_output.get("opening_signal", {}).get("direction", "unknown"),
                "confidence": 0.0,
                "reasons": [f"Gate parsing error: {str(e)}"],
                "warnings": [],
                "execution": {
                    "entry_type": "market",
                    "entry_price": float(market_values.get("current_price", 0) or 0),
                    "stop_loss": 0.0,
                    "take_profits": [],
                    "risk_reward": 0.0,
                    "position_size_note": "n/a"
                },
                "checks": {
                    "invalidation_triggered": bool(invalidation_triggered),
                    "checklist_score": {"met": 0, "total": 0},
                    "context_alignment": "weak"
                }
            }

    def execute_full_analysis(self, image_path: str = None, image_data: bytes = None) -> Dict[str, Any]:
        """Execute complete trading analysis pipeline.
        
        Args:
            image_path: Path to chart image (optional if image_data provided)
            image_data: Raw image bytes (optional if image_path provided)
            
        Returns:
            Complete analysis result with status and recommendations
        """
        try:
            # Step 1: Analyze chart with LLM
            llm_output = self.analyze_trading_chart(image_path, image_data)
            
            # Step 2: Save LLM output
            analysis_file = self.save_analysis_result(llm_output)
            
            # Step 3: Extract trading parameters
            symbol = llm_output.get('symbol', 'UNKNOWN')
            timeframe = self.validate_timeframe(llm_output.get('timeframe', '1m'))
            
            # Step 4: Fetch market data
            df = self.fetch_market_dataframe(symbol, timeframe)
            
            if df.empty:
                return {
                    "status": "error",
                    "message": f"No market data available for {symbol}",
                    "llm_output": llm_output,
                    "analysis_file": analysis_file
                }
            
            # Step 5: Calculate indicators
            df = self.calculate_all_indicators(df)
            
            # Step 6: Validate signal
            signal_valid, signal_status, triggered_conditions, market_values = self.validate_trading_signal(df, llm_output)
            
            result = {
                "status": "success",
                "signal_status": signal_status,
                "signal_valid": signal_valid,
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": market_values.get("current_price", 0),
                "current_rsi": market_values.get("current_rsi", 0),
                "triggered_conditions": triggered_conditions,
                "llm_output": llm_output,
                "analysis_file": analysis_file,
                "market_values": market_values
            }
            
            # Step 7: If signal is valid, run trade gate
            if signal_valid and signal_status == "valid":
                gate_result = self.llm_trade_gate_decision(
                    llm_output, market_values, True, False, triggered_conditions
                )
                
                gate_file = self.save_analysis_result(gate_result, "llm_gate")
                result.update({
                    "gate_result": gate_result,
                    "gate_file": gate_file,
                    "should_open": gate_result.get("should_open", False)
                })
                
                # Send notifications if trade approved
                if gate_result.get("should_open"):
                    try:
                        trade_data = {
                            "symbol": symbol,
                            "direction": gate_result.get("direction", "unknown"),
                            "current_price": gate_result.get("execution", {}).get("entry_price", 0),
                            "confidence": gate_result.get("confidence", 0),
                            "current_rsi": market_values.get("current_rsi", 0),
                            "stop_loss": gate_result.get("execution", {}).get("stop_loss", 0),
                            "risk_reward": gate_result.get("execution", {}).get("risk_reward", 0),
                            "take_profits": gate_result.get("execution", {}).get("take_profits", [])
                        }
                        notify_valid_trade(trade_data)
                        result["notification_sent"] = True
                    except Exception as e:
                        result["notification_error"] = str(e)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__
            }