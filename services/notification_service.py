import os
import json
import html
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NotificationService:
    """Service for sending trade notifications via Telegram"""
    
    def __init__(self):
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)
        
    def send_trade_notification(self, trade_data: Dict[str, Any], notification_type: str = "valid_trade") -> Dict[str, bool]:
        """
        Send trade notification using all available methods
        
        Args:
            trade_data: Dictionary containing trade information
            notification_type: Type of notification (valid_trade, invalidated, etc.)
            
        Returns:
            Dictionary with success status for each notification method
        """
        results = {
            "telegram": False
        }
        
        # Prepare notification message
        message = self._format_trade_message(trade_data, notification_type)
        
        # Try Telegram
        if self.telegram_bot_token and self.telegram_chat_id:
            results["telegram"] = self._send_telegram_notification(message, trade_data)
        
        return results
    
    def _format_trade_message(self, trade_data: Dict[str, Any], notification_type: str) -> str:
        """Format trade data into a readable notification message"""
        
        if notification_type == "valid_trade":
            title = "🚀 VALID TRADE SIGNAL"
            symbol = trade_data.get("symbol", "Unknown")
            direction = trade_data.get("direction", "Unknown")
            price = trade_data.get("current_price", 0)
            confidence = trade_data.get("confidence", 0)
            rsi = trade_data.get("current_rsi", 0)
            
            stop_loss = trade_data.get("stop_loss")
            risk_reward = trade_data.get("risk_reward")
            take_profits = trade_data.get("take_profits") or []
            
            message_lines = [
                title,
                "",
                f"📊 Symbol: {symbol}",
                f"📈 Direction: {direction.upper()}",
                f"💰 Price: ${price:.4f}",
                f"📊 RSI: {rsi:.2f}",
                f"🎯 Confidence: {confidence:.1%}",
            ]
            
            if stop_loss is not None:
                try:
                    message_lines.append(f"🛡️ Stop Loss: ${float(stop_loss):.4f}")
                except (TypeError, ValueError):
                    message_lines.append(f"🛡️ Stop Loss: {stop_loss}")
            
            if take_profits:
                message_lines.append("🎯 Take Profits:")
                for idx, tp in enumerate(take_profits, 1):
                    tp_price = tp.get("price")
                    tp_rr = tp.get("rr")
                    if tp_price is None:
                        continue
                    try:
                        tp_line = f"  • TP{idx}: ${float(tp_price):.4f}"
                    except (TypeError, ValueError):
                        tp_line = f"  • TP{idx}: {tp_price}"
                    if tp_rr:
                        tp_line += f" (R:R {tp_rr})"
                    message_lines.append(tp_line)
            
            if risk_reward is not None:
                message_lines.append(f"📐 Plan R:R: {risk_reward}")
            
            # Add multi-model comparison section if available
            multi_model_info = trade_data.get("_multi_model_comparison", {})
            if multi_model_info:
                message_lines.extend([
                    "",
                    "━━━━━━━━━━━━━━━━━━━━",
                    "🔬 MULTI-MODEL ANALYSIS",
                    "━━━━━━━━━━━━━━━━━━━━",
                    ""
                ])
                
                selected_model = multi_model_info.get("selected_model", "Unknown")
                selection_reason = multi_model_info.get("selection_reason", "N/A")
                all_results = multi_model_info.get("all_results", {})
                comparison_summary = multi_model_info.get("comparison_summary", {})
                errors = multi_model_info.get("errors", {})
                
                message_lines.append(f"Model Used: {selected_model}")
                message_lines.append(f"Selection: {selection_reason}")
                message_lines.append("")
                
                if all_results:
                    message_lines.append("All Models Results:")
                    for model_name, model_data in all_results.items():
                        model_direction = model_data.get("direction", "unknown").upper()
                        model_confidence = model_data.get("confidence", 0)
                        model_time = model_data.get("elapsed_time", 0)
                        model_stop_loss = model_data.get("stop_loss")
                        model_take_profits = model_data.get("take_profits", [])
                        direction_emoji = "🟢" if model_direction == "LONG" else "🔴" if model_direction == "SHORT" else "⚪"
                        selected_marker = " ⭐" if model_name == selected_model else ""
                        
                        message_lines.append(f"  • {direction_emoji} {model_name}{selected_marker}")
                        message_lines.append(f"    └ Direction: {model_direction}, Confidence: {model_confidence:.0%}, Time: {model_time:.1f}s")
                        
                        # Add stop loss if available
                        if model_stop_loss is not None:
                            try:
                                message_lines.append(f"    └ Stop Loss: ${float(model_stop_loss):.2f}")
                            except (TypeError, ValueError):
                                message_lines.append(f"    └ Stop Loss: {model_stop_loss}")
                        
                        # Add take profits if available
                        if model_take_profits:
                            tp_list = []
                            for idx, tp in enumerate(model_take_profits, 1):
                                # Handle both dict format (with rr) and simple price format
                                if isinstance(tp, dict):
                                    tp_price = tp.get("price")
                                    tp_rr = tp.get("rr")
                                else:
                                    tp_price = tp
                                    tp_rr = None
                                
                                try:
                                    tp_str = f"TP{idx}: ${float(tp_price):.2f}"
                                    if tp_rr is not None:
                                        tp_str += f" (R:R {tp_rr:.2f})"
                                    tp_list.append(tp_str)
                                except (TypeError, ValueError):
                                    tp_list.append(f"TP{idx}: {tp_price}")
                            if tp_list:
                                message_lines.append(f"    └ Take Profits: {', '.join(tp_list)}")
                        
                        # Add max RR ratio if available
                        model_max_rr = model_data.get("max_rr")
                        if model_max_rr is not None:
                            try:
                                message_lines.append(f"    └ Max R:R: {float(model_max_rr):.2f}")
                            except (TypeError, ValueError):
                                pass
                
                if errors:
                    message_lines.append("")
                    message_lines.append("Failed Models:")
                    for model_name, error_info in errors.items():
                        error_msg = error_info.get("error", "Unknown error")
                        # Show helpful error messages, but truncate very long ones
                        if len(error_msg) > 100:
                            error_msg = error_msg[:97] + "..."
                        message_lines.append(f"  • ❌ {model_name}: {error_msg}")
                
                # Add consensus info
                consensus = comparison_summary.get("consensus_direction")
                agreement = comparison_summary.get("agreement", False)
                if consensus:
                    agreement_emoji = "✅" if agreement else "⚠️"
                    consensus_text = f"{agreement_emoji} Consensus: {consensus.upper()}"
                    if not agreement:
                        consensus_text += " (models disagree)"
                    message_lines.append("")
                    message_lines.append(consensus_text)
            
            message_lines.extend([
                "",
                "✅ Trade approved by AI gate",
                f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
                
            ])
            
            message = "\n".join(message_lines)
            
        elif notification_type == "invalidated":
            title = "❌ TRADE INVALIDATED"
            symbol = trade_data.get("symbol", "Unknown")
            price = trade_data.get("current_price", 0)
            triggered_conditions = trade_data.get("triggered_conditions", [])
            
            message = f"""
{title}

📊 Symbol: {symbol}
💰 Price: ${price:.4f}
⚠️ Triggered Conditions: {', '.join(triggered_conditions) if triggered_conditions else 'None'}

❌ Trade signal no longer valid
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
        elif notification_type == "rejected":
            title = "❌ TRADE REJECTED BY GATE"
            symbol = trade_data.get("symbol", "Unknown")
            direction = trade_data.get("direction", "Unknown")
            price = trade_data.get("current_price", 0)
            confidence = trade_data.get("confidence", 0.0)
            reasons = trade_data.get("reasons", [])
            warnings = trade_data.get("warnings", [])
            checks = trade_data.get("checks", {})
            
            message_lines = [
                f"<b>{title}</b>",
                "",
                f"📊 <b>Symbol:</b> {symbol}",
                f"📈 <b>Direction:</b> {direction.upper()}",
                f"💰 <b>Price:</b> ${price:.4f}",
                f"🎯 <b>Confidence:</b> {confidence:.1%}",
                ""
            ]
            
            # Add rejection reasons
            if reasons:
                message_lines.append("━━━━━━━━━━━━━━━━━━━━")
                message_lines.append("<b>🚫 REJECTION REASONS</b>")
                message_lines.append("━━━━━━━━━━━━━━━━━━━━")
                message_lines.append("")
                for i, reason in enumerate(reasons, 1):
                    message_lines.append(f"{i}. {reason}")
                message_lines.append("")
            
            # Add warnings if any
            if warnings:
                message_lines.append("━━━━━━━━━━━━━━━━━━━━")
                message_lines.append("<b>⚠️ WARNINGS</b>")
                message_lines.append("━━━━━━━━━━━━━━━━━━━━")
                message_lines.append("")
                for i, warning in enumerate(warnings, 1):
                    message_lines.append(f"{i}. {warning}")
                message_lines.append("")
            
            # Add check details
            if checks:
                invalidation_triggered = checks.get("invalidation_triggered", False)
                checklist_score = checks.get("checklist_score", {})
                context_alignment = checks.get("context_alignment", "unknown")
                
                message_lines.append("━━━━━━━━━━━━━━━━━━━━")
                message_lines.append("<b>📋 VALIDATION CHECKS</b>")
                message_lines.append("━━━━━━━━━━━━━━━━━━━━")
                message_lines.append("")
                message_lines.append(f"• <b>Invalidation Triggered:</b> {'Yes' if invalidation_triggered else 'No'}")
                if checklist_score:
                    met = checklist_score.get("met", 0)
                    total = checklist_score.get("total", 0)
                    message_lines.append(f"• <b>Checklist Score:</b> {met}/{total}")
                message_lines.append(f"• <b>Context Alignment:</b> {context_alignment}")
                message_lines.append("")
            
            message_lines.append(f"⏰ <i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>")
            
            message = "\n".join(message_lines)
            
        else:
            title = "📱 TRADE UPDATE"
            message = f"""
{title}

{json.dumps(trade_data, indent=2, default=str)}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
        
        return message
    
    def _send_telegram_notification(self, message: str, trade_data: Dict[str, Any]) -> bool:
        """Send notification via Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Telegram notification sent successfully")
            return True
            
        except Exception as e:
            print(f"❌ Telegram notification failed: {e}")
            return False
    
    def send_telegram_image(self, image_path: str, caption: str = "") -> bool:
        """Send image to Telegram with optional caption"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
            
            with open(image_path, 'rb') as image_file:
                files = {'photo': image_file}
                data = {
                    "chat_id": self.telegram_chat_id,
                    "caption": caption,
                    "parse_mode": "HTML"
                }
                
                response = requests.post(url, data=data, files=files, timeout=30)
                response.raise_for_status()
            
            print(f"✅ Telegram image sent successfully")
            return True
            
        except Exception as e:
            print(f"❌ Telegram image send failed: {e}")
            return False
    
    def send_telegram_initial_analysis(self, llm_output: Dict[str, Any]) -> bool:
        """Send initial LLM analysis results to Telegram (text message with comprehensive conditions)"""
        try:
            symbol = html.escape(str(llm_output.get("symbol", "Unknown")))
            timeframe = html.escape(str(llm_output.get("timeframe", "Unknown")))
            time_of_screenshot = html.escape(str(llm_output.get("time_of_screenshot", "Unknown")))
            
            # Check if this is a multi-model analysis
            multi_model_info = llm_output.get("_multi_model_comparison", {})
            is_multi_model = bool(multi_model_info)
            
            # Get opening signal info
            opening_signal = llm_output.get("opening_signal", {})
            direction = html.escape(str(opening_signal.get("direction", "Unknown")).upper())
            is_met = opening_signal.get("is_met", False)
            
            # Get risk management
            risk_mgmt = llm_output.get("risk_management", {})
            stop_loss_info = risk_mgmt.get("stop_loss", {})
            take_profits = risk_mgmt.get("take_profit", [])
            
            # Get pattern analysis
            patterns = llm_output.get("pattern_analysis", [])
            top_patterns = []
            if patterns:
                sorted_patterns = sorted(patterns, key=lambda x: x.get("confidence", 0), reverse=True)
                top_patterns = sorted_patterns[:3]
            
            # Get key technical indicators
            tech_indicators = llm_output.get("technical_indicators", {}) or llm_output.get("core_indicators", {})
            rsi_info = tech_indicators.get("RSI14", {})
            rsi = rsi_info.get("value", "N/A")
            rsi_signal = html.escape(str(rsi_info.get("signal", "N/A")))
            
            macd = tech_indicators.get("MACD12_26_9", {})
            macd_histogram = macd.get("histogram", "N/A") if isinstance(macd, dict) else "N/A"
            
            stoch = tech_indicators.get("STOCH14_3_3", {})
            stoch_k = stoch.get("k_percent", "N/A")
            stoch_signal = html.escape(str(stoch.get("signal", "N/A")))
            
            volume_info = tech_indicators.get("VOLUME", {})
            volume_ratio = volume_info.get("ratio", "N/A")
            volume_trend = html.escape(str(volume_info.get("trend", "N/A")))
            
            # Get support/resistance
            sup_res = llm_output.get("support_resistance", {})
            support = sup_res.get("support", "N/A")
            resistance = sup_res.get("resistance", "N/A")
            
            # Get validity assessment
            validity = llm_output.get("validity_assessment", {})
            alignment_score = validity.get("alignment_score", validity.get("core_alignment_score", "N/A"))
            validity_notes = html.escape(str(validity.get("notes", "")))
            
            # Format message - Part 1: Overview
            message = f"""
<b>🔍 COMPREHENSIVE CHART ANALYSIS</b>

━━━━━━━━━━━━━━━━━━━━
<b>📊 TRADE SETUP OVERVIEW</b>
━━━━━━━━━━━━━━━━━━━━

<b>Symbol:</b> {symbol}
<b>Timeframe:</b> {timeframe}
<b>Screenshot:</b> {time_of_screenshot}
<b>Direction:</b> {'🟢 ' + direction if direction == 'LONG' else '🔴 ' + direction}
<b>Signal Status:</b> {'✅ MET' if is_met else '⏳ PENDING'}
"""
            
            # Add multi-model comparison section if available
            if is_multi_model:
                all_results = multi_model_info.get("all_results", {})
                comparison_summary = multi_model_info.get("comparison_summary", {})
                selected_model = html.escape(str(multi_model_info.get("selected_model", "Unknown")))
                errors = multi_model_info.get("errors", {})
                
                selection_reason = html.escape(str(multi_model_info.get("selection_reason", "N/A")))
                message += f"""
━━━━━━━━━━━━━━━━━━━━
<b>🔬 MULTI-MODEL ANALYSIS</b>
━━━━━━━━━━━━━━━━━━━━

<b>Model Used for Pipeline:</b> {selected_model}
<i>Selection: {selection_reason}</i>

<b>All Models Results:</b>
<i>Review all results to determine which model performs best</i>
"""
                for model_name, model_data in all_results.items():
                    model_name_esc = html.escape(str(model_name))
                    model_direction = html.escape(str(model_data.get("direction", "unknown")).upper())
                    model_confidence = model_data.get("confidence", 0)
                    model_time = model_data.get("elapsed_time", 0)
                    model_stop_loss = model_data.get("stop_loss")
                    model_take_profits = model_data.get("take_profits", [])
                    direction_emoji = "🟢" if model_direction == "LONG" else "🔴" if model_direction == "SHORT" else "⚪"
                    selected_marker = " ⭐" if model_name == multi_model_info.get("selected_model") else ""
                    
                    # Safe formatting for confidence and time
                    conf_str = f"{model_confidence:.0%}" if isinstance(model_confidence, (int, float)) else "N/A"
                    time_str = f"{model_time:.1f}s" if isinstance(model_time, (int, float)) else "N/A"
                    
                    message += f"• {direction_emoji} <b>{model_name_esc}</b>{selected_marker}\n"
                    message += f"  └ Direction: {model_direction}, Confidence: {conf_str}, Time: {time_str}\n"
                    
                    # Add stop loss if available
                    if model_stop_loss is not None:
                        try:
                            message += f"  └ Stop Loss: ${float(model_stop_loss):,.2f}\n"
                        except (TypeError, ValueError):
                            # Escape if it's a string
                            message += f"  └ Stop Loss: {html.escape(str(model_stop_loss))}\n"
                    
                    # Add take profits if available
                    if model_take_profits:
                        tp_list = []
                        for idx, tp in enumerate(model_take_profits, 1):
                            # Handle both dict format (with rr) and simple price format
                            if isinstance(tp, dict):
                                tp_price = tp.get("price")
                                tp_rr = tp.get("rr")
                            else:
                                tp_price = tp
                                tp_rr = None
                            
                            try:
                                tp_str = f"TP{idx}: ${float(tp_price):,.2f}"
                                if tp_rr is not None:
                                    try:
                                        tp_str += f" (R:R {float(tp_rr):.2f})"
                                    except (TypeError, ValueError):
                                        pass
                                tp_list.append(tp_str)
                            except (TypeError, ValueError):
                                tp_list.append(f"TP{idx}: {html.escape(str(tp_price))}")
                        if tp_list:
                            message += f"  └ Take Profits: {', '.join(tp_list)}\n"
                    
                    # Add max RR ratio if available
                    model_max_rr = model_data.get("max_rr")
                    if model_max_rr is not None:
                        try:
                            message += f"  └ Max R:R: {float(model_max_rr):.2f}\n"
                        except (TypeError, ValueError):
                            pass
                
                if errors:
                    message += f"\n<b>Failed Models:</b>\n"
                    for model_name, error_info in errors.items():
                        model_name_esc = html.escape(str(model_name))
                        error_msg = html.escape(str(error_info.get("error", "Unknown error")))
                        # Truncate long error messages
                        if len(error_msg) > 50:
                            error_msg = error_msg[:47] + "..."
                        message += f"• ❌ {model_name_esc}: {error_msg}\n"
                
                # Add consensus info
                consensus = comparison_summary.get("consensus_direction")
                agreement = comparison_summary.get("agreement", False)
                if consensus:
                    agreement_emoji = "✅" if agreement else "⚠️"
                    message += f"\n<b>Consensus:</b> {agreement_emoji} {html.escape(str(consensus).upper())}"
                    if not agreement:
                        message += " (models disagree)"
                    message += "\n"
                
                message += "\n"
            
            # Add alignment score
            if alignment_score != "N/A" and alignment_score is not None:
                try:
                    alignment_val = float(alignment_score)
                    message += f"<b>Alignment Score:</b> {alignment_val * 100:.0f}%\n"
                except (ValueError, TypeError):
                    # If conversion fails, print as is
                    message += f"<b>Alignment Score:</b> {alignment_score}\n"
            
            message += "\n"
            
            # Part 2: Technical Indicators
            
            # Safe formatting for support/resistance
            try:
                if isinstance(support, (int, float)):
                    supp_str = f"${support:,.2f}"
                else:
                    supp_str = f"${html.escape(str(support))}"
            except Exception:
                supp_str = "N/A"
                
            try:
                if isinstance(resistance, (int, float)):
                    res_str = f"${resistance:,.2f}"
                else:
                    res_str = f"${html.escape(str(resistance))}"
            except Exception:
                res_str = "N/A"

            message += f"""━━━━━━━━━━━━━━━━━━━━
<b>📈 TECHNICAL INDICATORS</b>
━━━━━━━━━━━━━━━━━━━━

<b>RSI14:</b> {rsi}
  └ Signal: {rsi_signal}

<b>Stochastic (14,3,3):</b> {stoch_k}
  └ Signal: {stoch_signal}

<b>MACD Histogram:</b> {macd_histogram}

<b>Volume Ratio:</b> {volume_ratio}
  └ Trend: {volume_trend}

<b>Support:</b> {supp_str}
<b>Resistance:</b> {res_str}

"""
            
            # Part 3: Risk Management
            sl_price = stop_loss_info.get("price", 0)
            sl_basis = html.escape(str(stop_loss_info.get("basis", "N/A")))
            
            # Safe formatting for SL
            try:
                if isinstance(sl_price, (int, float)):
                    sl_str = f"${sl_price:,.2f}"
                else:
                    sl_str = f"${html.escape(str(sl_price))}" if sl_price is not None else "N/A"
            except Exception:
                sl_str = "N/A"
            
            message += f"""━━━━━━━━━━━━━━━━━━━━
<b>🛡️ RISK MANAGEMENT</b>
━━━━━━━━━━━━━━━━━━━━

<b>Stop Loss:</b> {sl_str}
  └ Basis: {sl_basis}

"""
            
            # Add take profit levels
            if take_profits:
                message += "<b>Take Profit Targets:</b>\n"
                for i, tp in enumerate(take_profits, 1):
                    tp_price = tp.get("price", 0)
                    tp_basis = html.escape(str(tp.get("basis", "N/A")))
                    tp_rr = tp.get("rr", "N/A")
                    
                    try:
                        if isinstance(tp_price, (int, float)):
                            tp_p_str = f"${tp_price:,.2f}"
                        else:
                            tp_p_str = f"${html.escape(str(tp_price))}" if tp_price is not None else "N/A"
                    except Exception:
                        tp_p_str = "N/A"
                        
                    message += f"  TP{i}: {tp_p_str} (R:R {tp_rr})\n"
                    message += f"    └ {tp_basis}\n"
            else:
                message += "<b>Take Profit:</b> To be determined\n"
            
            message += "\n"
            
            # Part 4: Entry Conditions Checklist
            checklist = opening_signal.get("checklist", []) or opening_signal.get("core_checklist", [])
            if checklist:
                message += f"""━━━━━━━━━━━━━━━━━━━━
<b>✅ ENTRY CONDITIONS ({len(checklist)} items)</b>
━━━━━━━━━━━━━━━━━━━━

"""
                # Group by category if available
                for i, condition in enumerate(checklist[:8], 1):  # Limit to 8 to avoid message length issues
                    cond_id = html.escape(str(condition.get("id", f"condition_{i}")))
                    indicator = html.escape(str(condition.get("indicator", "")))
                    comparator = html.escape(str(condition.get("comparator", "")))
                    value = html.escape(str(condition.get("value", "")))
                    cond_type = condition.get("type", "")
                    
                    # Format condition nicely
                    if indicator:
                        message += f"{i}. {indicator} {comparator} {value}\n"
                    elif cond_type == "candle_pattern":
                        pattern_name = html.escape(str(condition.get("pattern", cond_id)))
                        message += f"{i}. Pattern: {pattern_name}\n"
                    elif cond_type == "price_retest":
                        level = html.escape(str(condition.get("level", "key level")))
                        message += f"{i}. Retest of {level}\n"
                    else:
                        message += f"{i}. {cond_id.replace('_', ' ').title()}\n"
                
                if len(checklist) > 8:
                    message += f"\n<i>...and {len(checklist) - 8} more conditions</i>\n"
            
            message += "\n"
            
            # Part 5: Invalidation Rules
            invalidation = opening_signal.get("invalidation", [])
            if invalidation:
                message += f"""━━━━━━━━━━━━━━━━━━━━
<b>🚫 INVALIDATION RULES ({len(invalidation)} items)</b>
━━━━━━━━━━━━━━━━━━━━

<i>Trade is INVALID if any of these occur:</i>

"""
                for i, invalid in enumerate(invalidation[:5], 1):  # Limit to 5
                    inv_id = html.escape(str(invalid.get("id", f"rule_{i}")))
                    inv_type = invalid.get("type", "")
                    level = html.escape(str(invalid.get("level", "")))
                    indicator = html.escape(str(invalid.get("indicator", "")))
                    comparator = html.escape(str(invalid.get("comparator", "")))
                    value = html.escape(str(invalid.get("value", "")))
                    
                    if inv_type == "price_breach":
                        message += f"{i}. Price closes {comparator} {level}\n"
                    elif indicator:
                        message += f"{i}. {indicator} {comparator} {value}\n"
                    else:
                        message += f"{i}. {inv_id.replace('_', ' ').title()}\n"
                
                if len(invalidation) > 5:
                    message += f"\n<i>...and {len(invalidation) - 5} more rules</i>\n"
            
            message += "\n"
            
            # Part 6: Patterns
            if top_patterns:
                message += f"""━━━━━━━━━━━━━━━━━━━━
<b>🎯 CHART PATTERNS</b>
━━━━━━━━━━━━━━━━━━━━

"""
                for i, pattern in enumerate(top_patterns, 1):
                    pattern_name = html.escape(str(pattern.get("pattern", "Unknown")))
                    confidence = pattern.get("confidence", 0)
                    
                    try:
                        conf_str = f"({confidence:.0%})" if isinstance(confidence, (int, float)) else ""
                    except Exception:
                        conf_str = ""
                        
                    message += f"{i}. {pattern_name.replace('_', ' ').title()} {conf_str}\n"
                message += "\n"
            
            # Part 7: Validity Notes
            if validity_notes:
                message += f"""━━━━━━━━━━━━━━━━━━━━
<b>📋 ANALYSIS NOTES</b>
━━━━━━━━━━━━━━━━━━━━

{validity_notes}

"""
            
            # Part 8: Next Steps
            summary_actions = llm_output.get("summary_actions", [])
            message += f"""━━━━━━━━━━━━━━━━━━━━
<b>⏳ NEXT STEPS</b>
━━━━━━━━━━━━━━━━━━━━

"""
            
            if summary_actions:
                for action in summary_actions[:4]:  # Top 4 actions
                    message += f"• {html.escape(str(action))}\n"
                message += "\n"
            else:
                message += """• Fetching real-time market data...
• Validating conditions with live indicators...
• Running trade gate analysis...

"""
            
            message += """<i>📱 You'll receive another notification when validation completes. This may take several minutes depending on the timeframe.</i>

━━━━━━━━━━━━━━━━━━━━
"""
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Comprehensive initial analysis sent to Telegram successfully")
            return True
            
        except Exception as e:
            print(f"❌ Telegram initial analysis send failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def send_telegram_polling_start(self, llm_output: Dict[str, Any], timeframe: str, wait_seconds: int) -> bool:
        """Send notification when polling starts, showing what we're waiting for"""
        try:
            symbol = html.escape(str(llm_output.get("symbol", "Unknown")))
            opening_signal = llm_output.get("opening_signal", {})
            direction = html.escape(str(opening_signal.get("direction", "Unknown")).upper())
            
            # Get checklist conditions we're waiting for
            checklist = opening_signal.get("checklist", []) or opening_signal.get("core_checklist", [])
            secondary_checklist = opening_signal.get("secondary_checklist", [])
            all_checklist = checklist + secondary_checklist
            
            # Get invalidation conditions we're watching
            invalidation = opening_signal.get("invalidation", [])
            
            # Format wait time
            wait_minutes = wait_seconds / 60
            if wait_minutes < 1:
                wait_time_str = f"{wait_seconds} seconds"
            elif wait_minutes < 60:
                wait_time_str = f"{int(wait_minutes)} minutes"
            else:
                wait_hours = wait_minutes / 60
                wait_time_str = f"{wait_hours:.1f} hours"
            
            # Build message
            message = f"""
<b>⏳ POLLING STARTED</b>

━━━━━━━━━━━━━━━━━━━━
<b>📊 PROPOSED TRADE</b>
━━━━━━━━━━━━━━━━━━━━

<b>Symbol:</b> {symbol}
<b>Timeframe:</b> {html.escape(str(timeframe))}
<b>Direction:</b> {'🟢 ' + direction if direction == 'LONG' else '🔴 ' + direction if direction == 'SHORT' else direction}

━━━━━━━━━━━━━━━━━━━━
<b>⏱️ POLLING DETAILS</b>
━━━━━━━━━━━━━━━━━━━━

<b>Check Interval:</b> {wait_time_str}
<b>Status:</b> Waiting for entry conditions to be met

"""
            
            # Add what we're waiting for (checklist conditions)
            if all_checklist:
                message += f"""━━━━━━━━━━━━━━━━━━━━
<b>✅ WAITING FOR ({len(all_checklist)} conditions)</b>
━━━━━━━━━━━━━━━━━━━━

<i>These conditions must be met for the signal to be valid:</i>

"""
                for i, condition in enumerate(all_checklist[:6], 1):  # Limit to 6 to keep message concise
                    cond_id = html.escape(str(condition.get("id", f"condition_{i}")))
                    indicator = html.escape(str(condition.get("indicator", "")))
                    comparator = html.escape(str(condition.get("comparator", "")))
                    value = html.escape(str(condition.get("value", "")))
                    cond_type = condition.get("type", "")
                    
                    # Format condition nicely
                    if indicator:
                        message += f"{i}. {indicator} {comparator} {value}\n"
                    elif cond_type == "candle_pattern":
                        pattern_name = html.escape(str(condition.get("pattern", cond_id)))
                        message += f"{i}. Pattern: {pattern_name}\n"
                    elif cond_type == "price_retest":
                        level = html.escape(str(condition.get("level", "key level")))
                        message += f"{i}. Retest of {level}\n"
                    else:
                        message += f"{i}. {cond_id.replace('_', ' ').title()}\n"
                
                if len(all_checklist) > 6:
                    message += f"\n<i>...and {len(all_checklist) - 6} more conditions</i>\n"
            else:
                message += """━━━━━━━━━━━━━━━━━━━━
<b>✅ WAITING FOR</b>
━━━━━━━━━━━━━━━━━━━━

<i>Entry conditions to be validated with live market data...</i>

"""
            
            message += "\n"
            
            # Add what we're watching for (invalidation conditions)
            if invalidation:
                message += f"""━━━━━━━━━━━━━━━━━━━━
<b>🚫 WATCHING FOR ({len(invalidation)} rules)</b>
━━━━━━━━━━━━━━━━━━━━

<i>Trade will be INVALIDATED if any of these occur:</i>

"""
                for i, invalid in enumerate(invalidation[:4], 1):  # Limit to 4
                    inv_id = html.escape(str(invalid.get("id", f"rule_{i}")))
                    inv_type = invalid.get("type", "")
                    level = html.escape(str(invalid.get("level", "")))
                    indicator = html.escape(str(invalid.get("indicator", "")))
                    comparator = html.escape(str(invalid.get("comparator", "")))
                    value = html.escape(str(invalid.get("value", "")))
                    
                    if inv_type == "price_breach":
                        message += f"{i}. Price closes {comparator} {level}\n"
                    elif indicator:
                        message += f"{i}. {indicator} {comparator} {value}\n"
                    else:
                        message += f"{i}. {inv_id.replace('_', ' ').title()}\n"
                
                if len(invalidation) > 4:
                    message += f"\n<i>...and {len(invalidation) - 4} more rules</i>\n"
            
            message += f"""
━━━━━━━━━━━━━━━━━━━━

<i>📱 I'll check the market every {wait_time_str} and notify you when:
• All conditions are met ✅
• Signal is invalidated ❌
• Max polling cycles reached ⏱️</i>

━━━━━━━━━━━━━━━━━━━━
"""
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Polling start notification sent to Telegram successfully")
            return True
            
        except Exception as e:
            print(f"❌ Telegram polling start notification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def send_telegram_extraction_complete(self, extracted_data: Dict[str, Any], model_name: str = None) -> bool:
        """Send notification when image extraction is complete, showing patterns and extracted data"""
        try:
            symbol = html.escape(str(extracted_data.get("symbol", "Unknown")))
            timeframe = html.escape(str(extracted_data.get("timeframe", "Unknown")))
            patterns = extracted_data.get("patterns", [])
            
            # Build very short message
            message = f"<b>✅ Extraction Complete</b>\n\n"
            if model_name:
                # Extract display name from model name (e.g., "gpt-4o" -> "GPT-4o", "deepseek/deepseek-chat" -> "DeepSeek")
                display_name = model_name
                if "gpt" in model_name.lower() or "openai" in model_name.lower():
                    display_name = "ChatGPT"
                elif "deepseek" in model_name.lower():
                    display_name = "DeepSeek"
                elif "gemini" in model_name.lower():
                    display_name = "Gemini"
                message += f"<b>Model:</b> {html.escape(display_name)}\n"
            message += f"<b>{symbol}</b> {timeframe}\n\n"
            
            if patterns:
                message += "<b>Patterns:</b>\n"
                for pattern in patterns[:3]:  # Max 3 patterns
                    pattern_name = html.escape(str(pattern.get("pattern", "Unknown")))
                    confidence = pattern.get("confidence", 0)
                    confidence_pct = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A"
                    message += f"• {pattern_name.replace('_', ' ').title()} ({confidence_pct})\n"
            else:
                message += "<i>No patterns detected</i>\n"
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Extraction complete notification sent to Telegram successfully")
            return True
            
        except Exception as e:
            print(f"❌ Telegram extraction notification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def send_telegram_multi_model_extraction(self, all_extractions: Dict[str, Dict[str, Any]], symbol: str = None, timeframe: str = None) -> bool:
        """Send a single notification combining extraction data from all models"""
        try:
            symbol_esc = html.escape(str(symbol or "Unknown"))
            timeframe_esc = html.escape(str(timeframe or "Unknown"))
            
            message = f"<b>✅ Multi-Model Extraction Complete</b>\n\n"
            message += f"<b>{symbol_esc}</b> {timeframe_esc}\n\n"
            message += "━━━━━━━━━━━━━━━━━━━━\n"
            
            for model_display_name, extraction_data in all_extractions.items():
                model_name_esc = html.escape(str(model_display_name))
                
                # Check if this model had an error
                if extraction_data.get("error"):
                    error_msg = html.escape(str(extraction_data.get("error", "Unknown error")))
                    if len(error_msg) > 50:
                        error_msg = error_msg[:47] + "..."
                    message += f"\n❌ <b>{model_name_esc}</b>\n"
                    message += f"   Error: {error_msg}\n"
                    continue
                
                # Get the result data
                result = extraction_data.get("result", {})
                patterns = result.get("patterns", []) or result.get("pattern_analysis", [])
                direction = result.get("opening_signal", {}).get("direction", "unknown")
                confidence = result.get("validity_assessment", {}).get("core_alignment_score", 0)
                elapsed_time = extraction_data.get("elapsed_time", 0)
                
                # Direction emoji
                direction_upper = str(direction).upper()
                direction_emoji = "🟢" if direction_upper == "LONG" else "🔴" if direction_upper == "SHORT" else "⚪"
                
                # Format confidence
                if isinstance(confidence, (int, float)):
                    conf_str = f"{confidence:.0%}"
                else:
                    conf_str = "N/A"
                
                # Format time
                if isinstance(elapsed_time, (int, float)):
                    time_str = f"{elapsed_time:.1f}s"
                else:
                    time_str = "N/A"
                
                message += f"\n{direction_emoji} <b>{model_name_esc}</b>\n"
                message += f"   └ Direction: {html.escape(direction_upper)}, Confidence: {conf_str}, Time: {time_str}\n"
                
                # Add patterns
                if patterns:
                    pattern_strs = []
                    for pattern in patterns[:3]:  # Max 3 patterns
                        if isinstance(pattern, dict):
                            pattern_name = pattern.get("pattern", "Unknown")
                            pattern_conf = pattern.get("confidence", 0)
                            if isinstance(pattern_conf, (int, float)):
                                pattern_strs.append(f"{pattern_name} ({pattern_conf:.0%})")
                            else:
                                pattern_strs.append(pattern_name)
                        else:
                            pattern_strs.append(str(pattern))
                    if pattern_strs:
                        message += f"   └ Patterns: {', '.join(pattern_strs)}\n"
                else:
                    message += f"   └ Patterns: None detected\n"
                
                # Add stop loss and take profits if available
                risk_mgmt = result.get("risk_management", {})
                stop_loss = risk_mgmt.get("stop_loss", {})
                take_profits = risk_mgmt.get("take_profit", [])
                
                if stop_loss:
                    sl_price = stop_loss.get("price") if isinstance(stop_loss, dict) else stop_loss
                    if sl_price is not None:
                        try:
                            message += f"   └ Stop Loss: ${float(sl_price):,.2f}\n"
                        except (TypeError, ValueError):
                            message += f"   └ Stop Loss: {html.escape(str(sl_price))}\n"
                
                if take_profits:
                    tp_strs = []
                    for idx, tp in enumerate(take_profits[:3], 1):  # Max 3 TPs
                        if isinstance(tp, dict):
                            tp_price = tp.get("price")
                            tp_rr = tp.get("rr")
                        else:
                            tp_price = tp
                            tp_rr = None
                        
                        try:
                            if tp_price is not None:
                                tp_str = f"TP{idx}: ${float(tp_price):,.2f}"
                                if tp_rr is not None:
                                    tp_str += f" (R:R {float(tp_rr):.2f})"
                                tp_strs.append(tp_str)
                        except (TypeError, ValueError):
                            if tp_price is not None:
                                tp_strs.append(f"TP{idx}: {tp_price}")
                    
                    if tp_strs:
                        message += f"   └ Take Profits: {', '.join(tp_strs)}\n"
            
            message += "\n━━━━━━━━━━━━━━━━━━━━\n"
            message += "<i>📱 Full analysis will follow...</i>\n"
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Multi-model extraction notification sent to Telegram successfully")
            return True
            
        except Exception as e:
            print(f"❌ Telegram multi-model extraction notification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def send_telegram_analysis(self, image_path: str, analysis_data: Dict[str, Any]) -> bool:
        """Send trading chart image with analysis to Telegram"""
        try:
            # Format analysis data into readable caption
            symbol = html.escape(str(analysis_data.get("symbol", "Unknown")))
            timeframe = html.escape(str(analysis_data.get("timeframe", "Unknown")))
            direction = html.escape(str(analysis_data.get("direction", "Unknown")))
            confidence = analysis_data.get("confidence", 0)
            
            caption = f"""
<b>📊 Trading Chart Analysis</b>

<b>Symbol:</b> {symbol}
<b>Timeframe:</b> {timeframe}
<b>Direction:</b> {direction.upper()}
<b>Confidence:</b> {confidence:.1%}

<b>Analysis complete!</b>
            """.strip()
            
            # Send image with caption
            return self.send_telegram_image(image_path, caption)
            
        except Exception as e:
            print(f"❌ Telegram analysis send failed: {e}")
            return False
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification methods with a sample message"""
        test_data = {
            "symbol": "BTCUSDT",
            "direction": "long",
            "current_price": 50000.00,
            "confidence": 0.85,
            "current_rsi": 45.2,
            "triggered_conditions": []
        }
        
        print("🧪 Testing notification system...")
        results = self.send_trade_notification(test_data, "valid_trade")
        
        print(f"💬 Telegram: {'✅ Success' if results['telegram'] else '❌ Failed'}")
        
        return results

# Convenience function for easy integration
def notify_valid_trade(trade_data: Dict[str, Any]) -> Dict[str, bool]:
    """Convenience function to send valid trade notification"""
    service = NotificationService()
    return service.send_trade_notification(trade_data, "valid_trade")

def notify_invalidated_trade(trade_data: Dict[str, Any]) -> Dict[str, bool]:
    """Convenience function to send invalidated trade notification"""
    service = NotificationService()
    return service.send_trade_notification(trade_data, "invalidated")

def notify_rejected_trade(trade_data: Dict[str, Any]) -> Dict[str, bool]:
    """Convenience function to send rejected trade notification"""
    service = NotificationService()
    return service.send_trade_notification(trade_data, "rejected")

def send_image_to_telegram(image_path: str, caption: str = "") -> bool:
    """Convenience function to send image to Telegram"""
    service = NotificationService()
    return service.send_telegram_image(image_path, caption)

def send_initial_analysis_to_telegram(llm_output: Dict[str, Any]) -> bool:
    """Convenience function to send initial LLM analysis to Telegram"""
    service = NotificationService()
    return service.send_telegram_initial_analysis(llm_output)

def send_polling_start_to_telegram(llm_output: Dict[str, Any], timeframe: str, wait_seconds: int) -> bool:
    """Convenience function to send polling start notification to Telegram"""
    service = NotificationService()
    return service.send_telegram_polling_start(llm_output, timeframe, wait_seconds)

def send_analysis_to_telegram(image_path: str, analysis_data: Dict[str, Any]) -> bool:
    """Convenience function to send trading analysis to Telegram"""
    service = NotificationService()
    return service.send_telegram_analysis(image_path, analysis_data)

def send_extraction_complete_to_telegram(extracted_data: Dict[str, Any], model_name: str = None) -> bool:
    """Convenience function to send extraction completion notification to Telegram"""
    service = NotificationService()
    return service.send_telegram_extraction_complete(extracted_data, model_name)

def send_multi_model_extraction_to_telegram(all_extractions: Dict[str, Dict[str, Any]], symbol: str = None, timeframe: str = None) -> bool:
    """Convenience function to send combined multi-model extraction notification to Telegram"""
    service = NotificationService()
    return service.send_telegram_multi_model_extraction(all_extractions, symbol, timeframe)

def send_telegram_message(message: str) -> bool:
    """
    Convenience function to send a simple HTML message to Telegram.
    Used by position monitor for SL/TP notifications.
    """
    service = NotificationService()
    if not service.telegram_enabled:
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{service.telegram_bot_token}/sendMessage"
        data = {
            "chat_id": service.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Telegram message failed: {e}")
        return False

def test_notification_system() -> Dict[str, bool]:
    """Test the notification system"""
    service = NotificationService()
    return service.test_notifications()

if __name__ == "__main__":
    # Test the notification system
    test_notification_system()
