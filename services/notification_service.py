import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NotificationService:
    """Service for sending trade notifications to iPhone via multiple methods"""
    
    def __init__(self):
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.email_smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
        self.email_smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.email_username = os.getenv("EMAIL_USERNAME")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.email_to = os.getenv("EMAIL_TO")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
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
            "pushover": False,
            "email": False,
            "telegram": False
        }
        
        # Prepare notification message
        message = self._format_trade_message(trade_data, notification_type)
        
        # Try Pushover first (most reliable for iPhone)
        if self.pushover_token and self.pushover_user:
            results["pushover"] = self._send_pushover_notification(message, trade_data)
        
        # Try email as backup
        if self.email_username and self.email_password and self.email_to:
            results["email"] = self._send_email_notification(message, trade_data)
        
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
            
        else:
            title = "📱 TRADE UPDATE"
            message = f"""
{title}

{json.dumps(trade_data, indent=2, default=str)}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
        
        return message
    
    def _send_pushover_notification(self, message: str, trade_data: Dict[str, Any]) -> bool:
        """Send notification via Pushover (excellent for iPhone)"""
        try:
            url = "https://api.pushover.net/1/messages.json"
            
            # Determine priority based on notification type
            priority = 1  # High priority for valid trades
            if "invalidated" in message:
                priority = 0  # Normal priority for invalidated trades
            
            data = {
                "token": self.pushover_token,
                "user": self.pushover_user,
                "message": message,
                "title": "Trading Assistant",
                "priority": priority,
                "sound": "cashregister" if "VALID TRADE" in message else "pushover"
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Pushover notification sent successfully")
            return True
            
        except Exception as e:
            print(f"❌ Pushover notification failed: {e}")
            return False
    
    def _send_email_notification(self, message: str, trade_data: Dict[str, Any]) -> bool:
        """Send notification via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = self.email_to
            msg['Subject'] = "Trading Assistant - Trade Alert"
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to server and send email
            server = smtplib.SMTP(self.email_smtp_server, self.email_smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_username, self.email_to, text)
            server.quit()
            
            print(f"✅ Email notification sent successfully")
            return True
            
        except Exception as e:
            print(f"❌ Email notification failed: {e}")
            return False
    
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
            symbol = llm_output.get("symbol", "Unknown")
            timeframe = llm_output.get("timeframe", "Unknown")
            time_of_screenshot = llm_output.get("time_of_screenshot", "Unknown")
            
            # Get opening signal info
            opening_signal = llm_output.get("opening_signal", {})
            direction = opening_signal.get("direction", "Unknown").upper()
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
            rsi_signal = rsi_info.get("signal", "N/A")
            
            macd = tech_indicators.get("MACD12_26_9", {})
            macd_histogram = macd.get("histogram", "N/A") if isinstance(macd, dict) else "N/A"
            
            stoch = tech_indicators.get("STOCH14_3_3", {})
            stoch_k = stoch.get("k_percent", "N/A")
            stoch_signal = stoch.get("signal", "N/A")
            
            volume_info = tech_indicators.get("VOLUME", {})
            volume_ratio = volume_info.get("ratio", "N/A")
            volume_trend = volume_info.get("trend", "N/A")
            
            # Get support/resistance
            sup_res = llm_output.get("support_resistance", {})
            support = sup_res.get("support", "N/A")
            resistance = sup_res.get("resistance", "N/A")
            
            # Get validity assessment
            validity = llm_output.get("validity_assessment", {})
            alignment_score = validity.get("alignment_score", validity.get("core_alignment_score", "N/A"))
            validity_notes = validity.get("notes", "")
            
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
            
            # Add alignment score
            if alignment_score != "N/A":
                alignment_pct = float(alignment_score) * 100 if isinstance(alignment_score, (int, float)) else alignment_score
                message += f"<b>Alignment Score:</b> {alignment_pct:.0f}%\n"
            
            message += "\n"
            
            # Part 2: Technical Indicators
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

<b>Support:</b> ${support if isinstance(support, str) else f'{support:,.2f}'}
<b>Resistance:</b> ${resistance if isinstance(resistance, str) else f'{resistance:,.2f}'}

"""
            
            # Part 3: Risk Management
            sl_price = stop_loss_info.get("price", 0)
            sl_basis = stop_loss_info.get("basis", "N/A")
            
            message += f"""━━━━━━━━━━━━━━━━━━━━
<b>🛡️ RISK MANAGEMENT</b>
━━━━━━━━━━━━━━━━━━━━

<b>Stop Loss:</b> ${sl_price:,.2f}
  └ Basis: {sl_basis}

"""
            
            # Add take profit levels
            if take_profits:
                message += "<b>Take Profit Targets:</b>\n"
                for i, tp in enumerate(take_profits, 1):
                    tp_price = tp.get("price", 0)
                    tp_basis = tp.get("basis", "N/A")
                    tp_rr = tp.get("rr", "N/A")
                    message += f"  TP{i}: ${tp_price:,.2f} (R:R {tp_rr})\n"
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
                    cond_id = condition.get("id", f"condition_{i}")
                    indicator = condition.get("indicator", "")
                    comparator = condition.get("comparator", "")
                    value = condition.get("value", "")
                    cond_type = condition.get("type", "")
                    
                    # Format condition nicely
                    if indicator:
                        message += f"{i}. {indicator} {comparator} {value}\n"
                    elif cond_type == "candle_pattern":
                        pattern_name = condition.get("pattern", cond_id)
                        message += f"{i}. Pattern: {pattern_name}\n"
                    elif cond_type == "price_retest":
                        level = condition.get("level", "key level")
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
                    inv_id = invalid.get("id", f"rule_{i}")
                    inv_type = invalid.get("type", "")
                    level = invalid.get("level", "")
                    indicator = invalid.get("indicator", "")
                    comparator = invalid.get("comparator", "")
                    value = invalid.get("value", "")
                    
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
                    pattern_name = pattern.get("pattern", "Unknown")
                    confidence = pattern.get("confidence", 0)
                    message += f"{i}. {pattern_name.replace('_', ' ').title()} ({confidence:.0%})\n"
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
                    message += f"• {action}\n"
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
            symbol = llm_output.get("symbol", "Unknown")
            opening_signal = llm_output.get("opening_signal", {})
            direction = opening_signal.get("direction", "Unknown").upper()
            
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
<b>Timeframe:</b> {timeframe}
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
                    cond_id = condition.get("id", f"condition_{i}")
                    indicator = condition.get("indicator", "")
                    comparator = condition.get("comparator", "")
                    value = condition.get("value", "")
                    cond_type = condition.get("type", "")
                    
                    # Format condition nicely
                    if indicator:
                        message += f"{i}. {indicator} {comparator} {value}\n"
                    elif cond_type == "candle_pattern":
                        pattern_name = condition.get("pattern", cond_id)
                        message += f"{i}. Pattern: {pattern_name}\n"
                    elif cond_type == "price_retest":
                        level = condition.get("level", "key level")
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
                    inv_id = invalid.get("id", f"rule_{i}")
                    inv_type = invalid.get("type", "")
                    level = invalid.get("level", "")
                    indicator = invalid.get("indicator", "")
                    comparator = invalid.get("comparator", "")
                    value = invalid.get("value", "")
                    
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
    
    def send_telegram_analysis(self, image_path: str, analysis_data: Dict[str, Any]) -> bool:
        """Send trading chart image with analysis to Telegram"""
        try:
            # Format analysis data into readable caption
            symbol = analysis_data.get("symbol", "Unknown")
            timeframe = analysis_data.get("timeframe", "Unknown")
            direction = analysis_data.get("direction", "Unknown")
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
        
        print(f"📱 Pushover: {'✅ Success' if results['pushover'] else '❌ Failed'}")
        print(f"📧 Email: {'✅ Success' if results['email'] else '❌ Failed'}")
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

def test_notification_system() -> Dict[str, bool]:
    """Test the notification system"""
    service = NotificationService()
    return service.test_notifications()

if __name__ == "__main__":
    # Test the notification system
    test_notification_system()
