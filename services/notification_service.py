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
            "email": False
        }
        
        # Prepare notification message
        message = self._format_trade_message(trade_data, notification_type)
        
        # Try Pushover first (most reliable for iPhone)
        if self.pushover_token and self.pushover_user:
            results["pushover"] = self._send_pushover_notification(message, trade_data)
        
        # Try email as backup
        if self.email_username and self.email_password and self.email_to:
            results["email"] = self._send_email_notification(message, trade_data)
        
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
            
            message = f"""
{title}

📊 Symbol: {symbol}
📈 Direction: {direction.upper()}
💰 Price: ${price:.4f}
📊 RSI: {rsi:.2f}
🎯 Confidence: {confidence:.1%}

✅ Trade approved by AI gate
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Check your trading platform!
            """.strip()
            
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

def test_notification_system() -> Dict[str, bool]:
    """Test the notification system"""
    service = NotificationService()
    return service.test_notifications()

if __name__ == "__main__":
    # Test the notification system
    test_notification_system()
