"""LLM service for chart analysis and trade gate decisions."""

import os
import json
import base64
from typing import Dict, Any
from openai import OpenAI
from prompt import OPENAI_VISION_PROMPT, TRADE_GATE_PROMPT


class LLMService:
    """Service for handling LLM interactions for chart analysis and trade decisions."""
    
    def __init__(self):
        """Initialize the LLM service with OpenAI client."""
        self.client = OpenAI()
    
    def analyze_trading_chart(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a trading chart image using OpenAI Vision.
        
        Args:
            image_path: Path to the chart image file
            
        Returns:
            Dictionary containing the LLM analysis results
        """
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": OPENAI_VISION_PROMPT},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def llm_trade_gate_decision(
        self,
        base_llm_output: Dict[str, Any],
        market_values: Dict[str, Any],
        checklist_passed: bool,
        invalidation_triggered: bool,
        triggered_conditions: list
    ) -> Dict[str, Any]:
        """
        Make a trade gate decision using LLM after programmatic validation.
        
        Args:
            base_llm_output: Original LLM analysis output
            market_values: Current market data values
            checklist_passed: Whether checklist conditions were met
            invalidation_triggered: Whether any invalidation conditions were triggered
            triggered_conditions: List of triggered invalidation conditions
            
        Returns:
            Dictionary containing the trade gate decision
        """
        # Prepare concise context for the gate
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
            model="gpt-4o-2024-08-06",
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
