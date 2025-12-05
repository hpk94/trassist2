#!/usr/bin/env python3
"""
Model Backtesting Script
Analyzes multi_model_comparison files and simulates trades to determine best performing model.
Uses 30x leverage with SL/TP from the model predictions.
"""

import json
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import requests
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
import time

# Configuration
LEVERAGE = 30
INITIAL_CAPITAL = 10000  # Starting capital in USD
POSITION_SIZE_PCT = 0.10  # 10% of capital per trade
MAX_TRADE_DURATION_MINUTES = 60  # Max trade duration before force close
DEFAULT_SL_PCT = 0.003  # 0.3% default SL if not provided
DEFAULT_TP_PCT = 0.006  # 0.6% default TP if not provided (2:1 R:R)


@dataclass
class TradeResult:
    model: str
    timestamp: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl_pct: float
    pnl_usd: float
    exit_reason: str  # 'sl', 'tp', 'timeout', 'error'
    duration_minutes: float


def fetch_binance_klines(symbol: str, start_time: datetime, limit: int = 500, interval: str = "1m") -> List:
    """Fetch klines from Binance API starting from a specific timestamp."""
    url = "https://api.binance.com/api/v3/klines"
    
    # Convert datetime to milliseconds
    start_ms = int(start_time.timestamp() * 1000)
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return []


def klines_to_dataframe(klines: List) -> pd.DataFrame:
    """Convert Binance klines to DataFrame."""
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    return df


def parse_multi_model_file(filepath: str) -> Dict[str, Any]:
    """Parse a multi_model_comparison JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_model_predictions(data: Dict) -> List[Dict]:
    """Extract predictions from each model in the comparison file."""
    predictions = []
    
    timestamp = data.get('timestamp')
    symbol = data.get('symbol', 'BTCUSDT')
    
    results = data.get('results', {})
    
    for model_name, model_data in results.items():
        try:
            result = model_data.get('result', {})
            opening_signal = result.get('opening_signal', {})
            risk_mgmt = result.get('risk_management', {})
            
            direction = opening_signal.get('direction', 'neutral')
            
            # Skip neutral predictions
            if direction == 'neutral':
                continue
            
            # Get SL
            sl_data = risk_mgmt.get('stop_loss', {})
            stop_loss = sl_data.get('price') if isinstance(sl_data, dict) else None
            
            # Get first TP
            tp_data = risk_mgmt.get('take_profit', [])
            take_profit = None
            if isinstance(tp_data, list) and len(tp_data) > 0:
                take_profit = tp_data[0].get('price')
            
            predictions.append({
                'model': model_name,
                'timestamp': timestamp,
                'symbol': symbol,
                'direction': direction,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
        except Exception as e:
            print(f"Error parsing {model_name}: {e}")
            continue
    
    return predictions


def simulate_trade(
    prediction: Dict,
    price_data: pd.DataFrame,
    leverage: int = LEVERAGE
) -> Optional[TradeResult]:
    """Simulate a single trade based on prediction and price data."""
    
    if price_data.empty:
        return None
    
    direction = prediction['direction']
    model = prediction['model']
    timestamp = prediction['timestamp']
    
    # Entry price is the close of the first candle
    entry_price = float(price_data.iloc[0]['close'])
    
    stop_loss = prediction.get('stop_loss')
    take_profit = prediction.get('take_profit')
    
    # If SL/TP not provided, calculate defaults
    if stop_loss is None or stop_loss <= 0:
        if direction == 'long':
            stop_loss = entry_price * (1 - DEFAULT_SL_PCT)
        else:
            stop_loss = entry_price * (1 + DEFAULT_SL_PCT)
    
    if take_profit is None or take_profit <= 0:
        if direction == 'long':
            take_profit = entry_price * (1 + DEFAULT_TP_PCT)
        else:
            take_profit = entry_price * (1 - DEFAULT_TP_PCT)
    
    # Validate SL/TP logic
    if direction == 'long':
        if stop_loss >= entry_price:
            stop_loss = entry_price * (1 - DEFAULT_SL_PCT)
        if take_profit <= entry_price:
            take_profit = entry_price * (1 + DEFAULT_TP_PCT)
    else:  # short
        if stop_loss <= entry_price:
            stop_loss = entry_price * (1 + DEFAULT_SL_PCT)
        if take_profit >= entry_price:
            take_profit = entry_price * (1 - DEFAULT_TP_PCT)
    
    exit_price = None
    exit_reason = 'timeout'
    duration_minutes = 0
    
    # Simulate through candles
    for idx, row in price_data.iterrows():
        if idx == 0:
            continue  # Skip entry candle
        
        high = float(row['high'])
        low = float(row['low'])
        close = float(row['close'])
        
        duration_minutes = idx  # Each candle is 1 minute
        
        if direction == 'long':
            # Check SL first (worst case)
            if low <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'sl'
                break
            # Check TP
            if high >= take_profit:
                exit_price = take_profit
                exit_reason = 'tp'
                break
        else:  # short
            # Check SL first
            if high >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'sl'
                break
            # Check TP
            if low <= take_profit:
                exit_price = take_profit
                exit_reason = 'tp'
                break
        
        # Max duration check
        if duration_minutes >= MAX_TRADE_DURATION_MINUTES:
            exit_price = close
            exit_reason = 'timeout'
            break
    
    # If we went through all data without exit, use last close
    if exit_price is None:
        exit_price = float(price_data.iloc[-1]['close'])
        exit_reason = 'timeout'
        duration_minutes = len(price_data) - 1
    
    # Calculate PnL
    if direction == 'long':
        pnl_pct = ((exit_price - entry_price) / entry_price) * leverage * 100
    else:  # short
        pnl_pct = ((entry_price - exit_price) / entry_price) * leverage * 100
    
    position_value = INITIAL_CAPITAL * POSITION_SIZE_PCT
    pnl_usd = position_value * (pnl_pct / 100)
    
    return TradeResult(
        model=model,
        timestamp=timestamp,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        pnl_pct=pnl_pct,
        pnl_usd=pnl_usd,
        exit_reason=exit_reason,
        duration_minutes=duration_minutes
    )


def analyze_model_performance(trades: List[TradeResult]) -> Dict:
    """Analyze performance metrics for a model."""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl_pct': 0,
            'total_pnl_usd': 0,
            'avg_pnl_pct': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'profit_factor': 0,
            'sl_hits': 0,
            'tp_hits': 0,
            'timeouts': 0,
            'max_drawdown_pct': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0
        }
    
    winning = [t for t in trades if t.pnl_pct > 0]
    losing = [t for t in trades if t.pnl_pct <= 0]
    
    total_wins = sum(t.pnl_usd for t in winning) if winning else 0
    total_losses = abs(sum(t.pnl_usd for t in losing)) if losing else 0
    
    # Calculate max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t.pnl_pct
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': len(winning) / len(trades) * 100 if trades else 0,
        'total_pnl_pct': sum(t.pnl_pct for t in trades),
        'total_pnl_usd': sum(t.pnl_usd for t in trades),
        'avg_pnl_pct': sum(t.pnl_pct for t in trades) / len(trades) if trades else 0,
        'avg_win_pct': sum(t.pnl_pct for t in winning) / len(winning) if winning else 0,
        'avg_loss_pct': sum(t.pnl_pct for t in losing) / len(losing) if losing else 0,
        'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
        'sl_hits': len([t for t in trades if t.exit_reason == 'sl']),
        'tp_hits': len([t for t in trades if t.exit_reason == 'tp']),
        'timeouts': len([t for t in trades if t.exit_reason == 'timeout']),
        'max_drawdown_pct': max_dd,
        'best_trade_pct': max(t.pnl_pct for t in trades),
        'worst_trade_pct': min(t.pnl_pct for t in trades),
        'avg_duration_min': sum(t.duration_minutes for t in trades) / len(trades)
    }


def main():
    print("=" * 80)
    print("MODEL BACKTESTING - Finding the Best Predictive Model")
    print(f"Configuration: {LEVERAGE}x Leverage, ${INITIAL_CAPITAL} Capital, {POSITION_SIZE_PCT*100}% Position Size")
    print("=" * 80)
    print()
    
    # Find all multi_model_comparison files
    llm_outputs_dir = os.path.join(os.path.dirname(__file__), 'llm_outputs')
    pattern = os.path.join(llm_outputs_dir, 'multi_model_comparison_*.json')
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} multi_model_comparison files")
    print()
    
    # Collect all predictions
    all_predictions = []
    for filepath in files:
        try:
            data = parse_multi_model_file(filepath)
            predictions = extract_model_predictions(data)
            all_predictions.extend(predictions)
        except Exception as e:
            print(f"Error reading {os.path.basename(filepath)}: {e}")
    
    print(f"Extracted {len(all_predictions)} actionable predictions (non-neutral)")
    
    # Count predictions per model
    model_counts = defaultdict(int)
    for p in all_predictions:
        model_counts[p['model']] += 1
    
    print("\nPredictions per model:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count} predictions")
    print()
    
    # Simulate trades
    print("Fetching price data and simulating trades...")
    print("-" * 60)
    
    model_trades: Dict[str, List[TradeResult]] = defaultdict(list)
    
    # Cache price data to avoid repeated API calls for same timestamp
    price_cache: Dict[str, pd.DataFrame] = {}
    
    for i, pred in enumerate(all_predictions):
        try:
            # Parse timestamp
            ts_str = pred['timestamp']
            try:
                ts = datetime.fromisoformat(ts_str)
            except:
                ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
            
            # Round to minute for cache key
            cache_key = ts.strftime("%Y%m%d_%H%M")
            
            # Fetch price data (with caching)
            if cache_key not in price_cache:
                klines = fetch_binance_klines(pred['symbol'], ts, limit=MAX_TRADE_DURATION_MINUTES + 10)
                price_cache[cache_key] = klines_to_dataframe(klines)
                time.sleep(0.1)  # Rate limiting
            
            price_data = price_cache[cache_key]
            
            if price_data.empty:
                print(f"  [{i+1}/{len(all_predictions)}] No price data for {pred['model']} @ {ts_str[:16]}")
                continue
            
            # Simulate the trade
            result = simulate_trade(pred, price_data)
            
            if result:
                model_trades[pred['model']].append(result)
                emoji = "✅" if result.pnl_pct > 0 else "❌"
                print(f"  [{i+1}/{len(all_predictions)}] {emoji} {pred['model']}: {result.direction.upper()} "
                      f"@ {result.entry_price:.1f} → {result.exit_price:.1f} ({result.exit_reason}) "
                      f"= {result.pnl_pct:+.1f}%")
            
        except Exception as e:
            print(f"  [{i+1}/{len(all_predictions)}] Error simulating {pred['model']}: {e}")
    
    # Analyze results
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    model_stats = {}
    for model, trades in model_trades.items():
        model_stats[model] = analyze_model_performance(trades)
    
    # Sort by total PnL
    sorted_models = sorted(model_stats.items(), key=lambda x: -x[1]['total_pnl_pct'])
    
    print(f"{'Model':<15} {'Trades':>7} {'Win Rate':>10} {'Total PnL':>12} {'Avg PnL':>10} {'PF':>8} {'SL':>5} {'TP':>5} {'Max DD':>10}")
    print("-" * 95)
    
    for model, stats in sorted_models:
        print(f"{model:<15} {stats['total_trades']:>7} {stats['win_rate']:>9.1f}% "
              f"{stats['total_pnl_pct']:>+11.1f}% {stats['avg_pnl_pct']:>+9.1f}% "
              f"{stats['profit_factor']:>8.2f} {stats['sl_hits']:>5} {stats['tp_hits']:>5} "
              f"{stats['max_drawdown_pct']:>9.1f}%")
    
    print()
    print("=" * 80)
    print("DETAILED MODEL ANALYSIS")
    print("=" * 80)
    
    for model, stats in sorted_models:
        print(f"\n📊 {model}")
        print("-" * 40)
        print(f"  Total Trades:     {stats['total_trades']}")
        print(f"  Win Rate:         {stats['win_rate']:.1f}%")
        print(f"  Total PnL:        {stats['total_pnl_pct']:+.2f}% (${stats['total_pnl_usd']:+,.2f})")
        print(f"  Avg Trade PnL:    {stats['avg_pnl_pct']:+.2f}%")
        print(f"  Avg Win:          {stats['avg_win_pct']:+.2f}%")
        print(f"  Avg Loss:         {stats['avg_loss_pct']:+.2f}%")
        print(f"  Profit Factor:    {stats['profit_factor']:.2f}")
        print(f"  Best Trade:       {stats['best_trade_pct']:+.2f}%")
        print(f"  Worst Trade:      {stats['worst_trade_pct']:+.2f}%")
        print(f"  Max Drawdown:     {stats['max_drawdown_pct']:.2f}%")
        print(f"  Exit by SL:       {stats['sl_hits']} ({stats['sl_hits']/stats['total_trades']*100:.1f}%)" if stats['total_trades'] > 0 else "  Exit by SL: N/A")
        print(f"  Exit by TP:       {stats['tp_hits']} ({stats['tp_hits']/stats['total_trades']*100:.1f}%)" if stats['total_trades'] > 0 else "  Exit by TP: N/A")
        print(f"  Exit by Timeout:  {stats['timeouts']} ({stats['timeouts']/stats['total_trades']*100:.1f}%)" if stats['total_trades'] > 0 else "  Exit by Timeout: N/A")
        if 'avg_duration_min' in stats:
            print(f"  Avg Duration:     {stats['avg_duration_min']:.1f} minutes")
    
    # Determine winner
    print()
    print("=" * 80)
    if sorted_models:
        winner = sorted_models[0]
        print(f"🏆 BEST MODEL: {winner[0]}")
        print(f"   Total PnL: {winner[1]['total_pnl_pct']:+.2f}% (${winner[1]['total_pnl_usd']:+,.2f})")
        print(f"   Win Rate: {winner[1]['win_rate']:.1f}%")
        print(f"   Profit Factor: {winner[1]['profit_factor']:.2f}")
    else:
        print("No trades to analyze")
    print("=" * 80)
    
    # Save detailed results
    output_file = os.path.join(llm_outputs_dir, 'backtest_results.json')
    results_data = {
        'config': {
            'leverage': LEVERAGE,
            'initial_capital': INITIAL_CAPITAL,
            'position_size_pct': POSITION_SIZE_PCT,
            'max_trade_duration_minutes': MAX_TRADE_DURATION_MINUTES,
            'default_sl_pct': DEFAULT_SL_PCT,
            'default_tp_pct': DEFAULT_TP_PCT
        },
        'files_analyzed': len(files),
        'total_predictions': len(all_predictions),
        'model_statistics': {m: s for m, s in model_stats.items()},
        'individual_trades': {
            model: [
                {
                    'timestamp': t.timestamp,
                    'direction': t.direction,
                    'entry': t.entry_price,
                    'exit': t.exit_price,
                    'sl': t.stop_loss,
                    'tp': t.take_profit,
                    'pnl_pct': t.pnl_pct,
                    'pnl_usd': t.pnl_usd,
                    'exit_reason': t.exit_reason,
                    'duration_min': t.duration_minutes
                }
                for t in trades
            ]
            for model, trades in model_trades.items()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

