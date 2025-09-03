"""Refactored main application using service architecture."""

from services.trading_orchestrator import TradingOrchestrator


def main():
    """Main application entry point."""
    # Initialize the trading orchestrator
    orchestrator = TradingOrchestrator()
    
    # Test image path
    test_image = "BTCUSDT.P_2025-09-02_22-55-40_545b4.png"
    
    # Run complete analysis
    print("Starting trading analysis...")
    results = orchestrator.run_complete_analysis(test_image, max_cycles=5)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Symbol: {results['llm_output']['symbol']}")
    print(f"Timeframe: {results['llm_output']['timeframe']}")
    print(f"Signal Status: {results['signal_status']}")
    print(f"Signal Valid: {results['signal_valid']}")
    
    if results['triggered_conditions']:
        print(f"Triggered Conditions: {', '.join(results['triggered_conditions'])}")
    
    if results['gate_result']:
        print(f"Gate Decision: {'APPROVED' if results['gate_result'].get('should_open') else 'REJECTED'}")
        if results['gate_result'].get('confidence'):
            print(f"Confidence: {results['gate_result']['confidence']:.2f}")


if __name__ == "__main__":
    main()
