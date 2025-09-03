"""
Database module for managing trade state persistence across devices.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

DATABASE_PATH = "trades.db"

class TradeDatabase:
    """Database manager for trade state persistence"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    llm_output TEXT NOT NULL,
                    signal_status TEXT,
                    signal_valid BOOLEAN,
                    market_values TEXT,
                    gate_result TEXT,
                    output_files TEXT,
                    notes TEXT
                )
            ''')
            
            # Create trade_history table for tracking changes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    FOREIGN KEY (trade_id) REFERENCES trades (trade_id)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def create_trade(self, trade_data: Dict[str, Any]) -> str:
        """Create a new trade record"""
        trade_id = f"trade_{int(datetime.now().timestamp())}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Deactivate any existing active trades
            cursor.execute('''
                UPDATE trades 
                SET status = 'replaced', updated_at = CURRENT_TIMESTAMP 
                WHERE status = 'active'
            ''')
            
            # Insert new trade
            cursor.execute('''
                INSERT INTO trades (
                    trade_id, symbol, timeframe, direction, llm_output,
                    signal_status, signal_valid, market_values, gate_result,
                    output_files, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id,
                trade_data.get('symbol'),
                trade_data.get('timeframe'),
                trade_data.get('direction'),
                json.dumps(trade_data.get('llm_output', {})),
                trade_data.get('signal_status'),
                trade_data.get('signal_valid'),
                json.dumps(trade_data.get('market_values', {})),
                json.dumps(trade_data.get('gate_result', {})),
                json.dumps(trade_data.get('output_files', {})),
                trade_data.get('notes', '')
            ))
            
            # Log the creation
            cursor.execute('''
                INSERT INTO trade_history (trade_id, action, details)
                VALUES (?, 'created', ?)
            ''', (trade_id, json.dumps({'created_by': 'system'})))
            
            conn.commit()
        
        return trade_id
    
    def get_active_trade(self) -> Optional[Dict[str, Any]]:
        """Get the currently active trade"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM trades 
                WHERE status = 'active' 
                ORDER BY created_at DESC 
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None
    
    def update_trade_status(self, trade_id: str, status: str, notes: str = None) -> bool:
        """Update trade status (active, completed, canceled, etc.)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update trade status
            cursor.execute('''
                UPDATE trades 
                SET status = ?, updated_at = CURRENT_TIMESTAMP, notes = COALESCE(?, notes)
                WHERE trade_id = ?
            ''', (status, notes, trade_id))
            
            if cursor.rowcount > 0:
                # Log the status change
                cursor.execute('''
                    INSERT INTO trade_history (trade_id, action, details)
                    VALUES (?, ?, ?)
                ''', (trade_id, f'status_changed_to_{status}', json.dumps({'notes': notes})))
                
                conn.commit()
                return True
            return False
    
    def cancel_active_trade(self, notes: str = None) -> bool:
        """Cancel the currently active trade"""
        active_trade = self.get_active_trade()
        if active_trade:
            return self.update_trade_status(active_trade['trade_id'], 'canceled', notes)
        return False
    
    def get_trade_history(self, trade_id: str = None) -> List[Dict[str, Any]]:
        """Get trade history for a specific trade or all trades"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if trade_id:
                cursor.execute('''
                    SELECT * FROM trade_history 
                    WHERE trade_id = ? 
                    ORDER BY timestamp DESC
                ''', (trade_id,))
            else:
                cursor.execute('''
                    SELECT * FROM trade_history 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                ''')
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_all_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all trades with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM trades 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary"""
        result = dict(row)
        
        # Parse JSON fields
        json_fields = ['llm_output', 'market_values', 'gate_result', 'output_files']
        for field in json_fields:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = {}
        
        return result
    
    def cleanup_old_trades(self, days_old: int = 30):
        """Clean up old completed/canceled trades"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM trades 
                WHERE status IN ('completed', 'canceled', 'replaced') 
                AND created_at < datetime('now', '-{} days')
            '''.format(days_old))
            
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count

# Global database instance
db = TradeDatabase()

# Convenience functions
def create_trade(trade_data: Dict[str, Any]) -> str:
    """Create a new trade and return trade ID"""
    return db.create_trade(trade_data)

def get_active_trade() -> Optional[Dict[str, Any]]:
    """Get the currently active trade"""
    return db.get_active_trade()

def cancel_active_trade(notes: str = None) -> bool:
    """Cancel the currently active trade"""
    return db.cancel_active_trade(notes)

def update_trade_status(trade_id: str, status: str, notes: str = None) -> bool:
    """Update trade status"""
    return db.update_trade_status(trade_id, status, notes)

def get_trade_history(trade_id: str = None) -> List[Dict[str, Any]]:
    """Get trade history"""
    return db.get_trade_history(trade_id)

def get_all_trades(limit: int = 50) -> List[Dict[str, Any]]:
    """Get all trades"""
    return db.get_all_trades(limit)
