#!/usr/bin/env python3
"""
Manual analysis test script
Tests the complete analysis pipeline on a specific image
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_manual_analysis.py <image_path>")
        print("\nExample:")
        print("  python test_manual_analysis.py uploads/test_chart_custom_name.png")
        print("\nThis will run the complete analysis pipeline and send notifications to Telegram.")
        return 1
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    print("=" * 70)
    print("MANUAL ANALYSIS TEST")
    print("=" * 70)
    print(f"\n📊 Image: {image_path}")
    print(f"📏 Size: {os.path.getsize(image_path) / 1024:.2f} KB")
    print("\n⏳ Starting analysis...")
    print("This may take several minutes depending on the timeframe.")
    print("\nYou should receive Telegram notifications:")
    print("  1. Immediately: Image upload confirmation")
    print("  2. After analysis: Trade signal or invalidation")
    print("\n" + "=" * 70 + "\n")
    
    try:
        # Import here to catch import errors early
        from web_app import run_trading_analysis
        
        # Run the analysis
        result = run_trading_analysis(image_path)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        
        if result.get("success"):
            print("✅ Analysis succeeded!")
            
            llm_output = result.get("llm_output", {})
            signal_status = result.get("signal_status", "unknown")
            gate_result = result.get("gate_result", {})
            
            print(f"\n📊 Symbol: {llm_output.get('symbol', 'Unknown')}")
            print(f"⏰ Timeframe: {llm_output.get('timeframe', 'Unknown')}")
            print(f"📈 Signal Status: {signal_status}")
            
            if gate_result:
                should_open = gate_result.get("should_open", False)
                direction = gate_result.get("direction", "unknown")
                confidence = gate_result.get("confidence", 0)
                
                print(f"\n🚪 Gate Decision:")
                print(f"   Should Open: {'YES ✅' if should_open else 'NO ❌'}")
                print(f"   Direction: {direction.upper()}")
                print(f"   Confidence: {confidence:.1%}")
                
                if should_open:
                    execution = gate_result.get("execution", {})
                    print(f"\n💰 Trade Details:")
                    print(f"   Entry: ${execution.get('entry_price', 0):.2f}")
                    print(f"   Stop Loss: ${execution.get('stop_loss', 0):.2f}")
                    print(f"   Risk/Reward: {execution.get('risk_reward', 0):.2f}")
                    
                    reasons = gate_result.get("reasons", [])
                    if reasons:
                        print(f"\n📝 Reasons:")
                        for reason in reasons[:5]:
                            print(f"   • {reason}")
                else:
                    reasons = gate_result.get("reasons", [])
                    if reasons:
                        print(f"\n❌ Rejection Reasons:")
                        for reason in reasons[:5]:
                            print(f"   • {reason}")
            
            output_files = result.get("output_files", {})
            if output_files.get("llm_output"):
                print(f"\n📄 Output saved to:")
                print(f"   {output_files['llm_output']}")
            
            print(f"\n📱 Telegram notifications should have been sent!")
            print("   Check your Telegram for the analysis results.")
            
        else:
            print("❌ Analysis failed!")
            error = result.get("error", "Unknown error")
            error_type = result.get("error_type", "Unknown")
            print(f"\n🔴 Error Type: {error_type}")
            print(f"🔴 Error Message: {error}")
            print("\n📱 An error notification should have been sent to Telegram.")
        
        print("\n" + "=" * 70 + "\n")
        return 0 if result.get("success") else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())



