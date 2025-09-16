"""Status tracking service for persistent trade analysis storage."""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager

@dataclass
class TradeAnalysis:
    """Data class for trade analysis records."""
    analysis_id: str
    user_id: int
    username: str
    symbol: str
    timeframe: str
    signal_status: str
    current_price: float
    current_rsi: Optional[float]
    timestamp: str
    llm_output: Dict[str, Any]
    gate_result: Optional[Dict[str, Any]] = None
    triggered_conditions: Optional[List[str]] = None
    market_values: Optional[Dict[str, Any]] = None
    analysis_file: Optional[str] = None
    gate_file: Optional[str] = None
    notification_sent: bool = False

class StatusService:
    """Service for tracking and storing trade analysis status."""
    
    def __init__(self, db_path: str = "trade_status.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_analyses (
                    analysis_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal_status TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    current_rsi REAL,
                    timestamp TEXT NOT NULL,
                    llm_output TEXT NOT NULL,
                    gate_result TEXT,
                    triggered_conditions TEXT,
                    market_values TEXT,
                    analysis_file TEXT,
                    gate_file TEXT,
                    notification_sent BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON trade_analyses(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trade_analyses(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_status ON trade_analyses(signal_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trade_analyses(timestamp)")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def save_analysis(self, analysis: TradeAnalysis) -> bool:
        """Save or update a trade analysis record.
        
        Args:
            analysis: TradeAnalysis object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                # Convert complex objects to JSON strings
                llm_output_json = json.dumps(analysis.llm_output)
                gate_result_json = json.dumps(analysis.gate_result) if analysis.gate_result else None
                triggered_conditions_json = json.dumps(analysis.triggered_conditions) if analysis.triggered_conditions else None
                market_values_json = json.dumps(analysis.market_values) if analysis.market_values else None
                
                conn.execute("""
                    INSERT OR REPLACE INTO trade_analyses 
                    (analysis_id, user_id, username, symbol, timeframe, signal_status, 
                     current_price, current_rsi, timestamp, llm_output, gate_result, 
                     triggered_conditions, market_values, analysis_file, gate_file, 
                     notification_sent, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    analysis.analysis_id, analysis.user_id, analysis.username,
                    analysis.symbol, analysis.timeframe, analysis.signal_status,
                    analysis.current_price, analysis.current_rsi, analysis.timestamp,
                    llm_output_json, gate_result_json, triggered_conditions_json,
                    market_values_json, analysis.analysis_file, analysis.gate_file,
                    analysis.notification_sent
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False
    
    def get_analysis(self, analysis_id: str) -> Optional[TradeAnalysis]:
        """Get a specific trade analysis by ID.
        
        Args:
            analysis_id: Unique analysis identifier
            
        Returns:
            TradeAnalysis object if found, None otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM trade_analyses WHERE analysis_id = ?",
                    (analysis_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_analysis(row)
                return None
        except Exception as e:
            print(f"Error getting analysis: {e}")
            return None
    
    def get_user_analyses(self, user_id: int, limit: int = 10, status_filter: str = None) -> List[TradeAnalysis]:
        """Get trade analyses for a specific user.
        
        Args:
            user_id: Telegram user ID
            limit: Maximum number of records to return
            status_filter: Optional status filter (valid, invalidated, pending, error)
            
        Returns:
            List of TradeAnalysis objects
        """
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM trade_analyses WHERE user_id = ?"
                params = [user_id]
                
                if status_filter:
                    query += " AND signal_status = ?"
                    params.append(status_filter)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_analysis(row) for row in rows]
        except Exception as e:
            print(f"Error getting user analyses: {e}")
            return []
    
    def get_active_analyses(self, user_id: int = None) -> List[TradeAnalysis]:
        """Get active trade analyses (valid or pending status).
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            List of active TradeAnalysis objects
        """
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM trade_analyses WHERE signal_status IN ('valid', 'pending')"
                params = []
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                query += " ORDER BY created_at DESC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_analysis(row) for row in rows]
        except Exception as e:
            print(f"Error getting active analyses: {e}")
            return []
    
    def update_analysis_status(self, analysis_id: str, new_status: str, 
                             triggered_conditions: List[str] = None,
                             market_values: Dict[str, Any] = None) -> bool:
        """Update the status of a trade analysis.
        
        Args:
            analysis_id: Analysis ID to update
            new_status: New signal status
            triggered_conditions: Optional list of triggered conditions
            market_values: Optional updated market values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                triggered_json = json.dumps(triggered_conditions) if triggered_conditions else None
                market_json = json.dumps(market_values) if market_values else None
                
                conn.execute("""
                    UPDATE trade_analyses 
                    SET signal_status = ?, triggered_conditions = ?, market_values = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE analysis_id = ?
                """, (new_status, triggered_json, market_json, analysis_id))
                
                conn.commit()
                return conn.total_changes > 0
        except Exception as e:
            print(f"Error updating analysis status: {e}")
            return False
    
    def mark_notification_sent(self, analysis_id: str) -> bool:
        """Mark that notification has been sent for an analysis.
        
        Args:
            analysis_id: Analysis ID to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE trade_analyses SET notification_sent = TRUE, updated_at = CURRENT_TIMESTAMP WHERE analysis_id = ?",
                    (analysis_id,)
                )
                conn.commit()
                return conn.total_changes > 0
        except Exception as e:
            print(f"Error marking notification sent: {e}")
            return False
    
    def cleanup_old_analyses(self, days_old: int = 30) -> int:
        """Remove old trade analyses to keep database clean.
        
        Args:
            days_old: Remove analyses older than this many days
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM trade_analyses WHERE created_at < ?",
                    (cutoff_str,)
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"Error cleaning up old analyses: {e}")
            return 0
    
    def get_statistics(self, user_id: int = None, days: int = 7) -> Dict[str, Any]:
        """Get statistics for trade analyses.
        
        Args:
            user_id: Optional user ID filter
            days: Number of days to include in statistics
            
        Returns:
            Dictionary with statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            with self._get_connection() as conn:
                query = "SELECT signal_status, COUNT(*) as count FROM trade_analyses WHERE created_at >= ?"
                params = [cutoff_str]
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                query += " GROUP BY signal_status"
                
                cursor = conn.execute(query, params)
                status_counts = {row['signal_status']: row['count'] for row in cursor.fetchall()}
                
                # Get total count
                total_query = "SELECT COUNT(*) as total FROM trade_analyses WHERE created_at >= ?"
                total_params = [cutoff_str]
                if user_id:
                    total_query += " AND user_id = ?"
                    total_params.append(user_id)
                
                cursor = conn.execute(total_query, total_params)
                total_count = cursor.fetchone()['total']
                
                return {
                    'total_analyses': total_count,
                    'status_breakdown': status_counts,
                    'success_rate': status_counts.get('valid', 0) / max(total_count, 1),
                    'invalidation_rate': status_counts.get('invalidated', 0) / max(total_count, 1),
                    'pending_rate': status_counts.get('pending', 0) / max(total_count, 1),
                    'error_rate': status_counts.get('error', 0) / max(total_count, 1),
                    'period_days': days
                }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def _row_to_analysis(self, row: sqlite3.Row) -> TradeAnalysis:
        """Convert database row to TradeAnalysis object."""
        return TradeAnalysis(
            analysis_id=row['analysis_id'],
            user_id=row['user_id'],
            username=row['username'],
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            signal_status=row['signal_status'],
            current_price=row['current_price'],
            current_rsi=row['current_rsi'],
            timestamp=row['timestamp'],
            llm_output=json.loads(row['llm_output']),
            gate_result=json.loads(row['gate_result']) if row['gate_result'] else None,
            triggered_conditions=json.loads(row['triggered_conditions']) if row['triggered_conditions'] else None,
            market_values=json.loads(row['market_values']) if row['market_values'] else None,
            analysis_file=row['analysis_file'],
            gate_file=row['gate_file'],
            notification_sent=bool(row['notification_sent'])
        )
    
    def save_analysis_from_dict(self, analysis_dict: Dict[str, Any]) -> bool:
        """Save analysis from dictionary format (for backward compatibility).
        
        Args:
            analysis_dict: Dictionary containing analysis data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            analysis = TradeAnalysis(
                analysis_id=analysis_dict.get('analysis_id', ''),
                user_id=analysis_dict.get('user_id', 0),
                username=analysis_dict.get('username', ''),
                symbol=analysis_dict.get('symbol', ''),
                timeframe=analysis_dict.get('timeframe', ''),
                signal_status=analysis_dict.get('signal_status', ''),
                current_price=analysis_dict.get('current_price', 0.0),
                current_rsi=analysis_dict.get('current_rsi'),
                timestamp=analysis_dict.get('timestamp', ''),
                llm_output=analysis_dict.get('llm_output', {}),
                gate_result=analysis_dict.get('gate_result'),
                triggered_conditions=analysis_dict.get('triggered_conditions'),
                market_values=analysis_dict.get('market_values'),
                analysis_file=analysis_dict.get('analysis_file'),
                gate_file=analysis_dict.get('gate_file'),
                notification_sent=analysis_dict.get('notification_sent', False)
            )
            return self.save_analysis(analysis)
        except Exception as e:
            print(f"Error saving analysis from dict: {e}")
            return False