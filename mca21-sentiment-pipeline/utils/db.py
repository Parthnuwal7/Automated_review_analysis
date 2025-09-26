"""SQLite database interface for storing processed reviews and results."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading

from config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for review analysis data."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DB_PATH
        self._connection = None
        self._lock = threading.Lock()
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"DatabaseManager initialized with database: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            
            # Configure connection
            conn.row_factory = sqlite3.Row  # Enable column access by name
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
            conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
            
            return conn
            
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            with self._get_connection() as conn:
                # Create reviews table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS reviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        review_id TEXT UNIQUE NOT NULL,
                        original_text TEXT NOT NULL,
                        translated_text TEXT,
                        language TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        sentiment_score REAL,
                        intent TEXT NOT NULL,
                        intent_score REAL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # Create aspects table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS aspects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        review_id TEXT NOT NULL,
                        aspect_id INTEGER,
                        aspect_label TEXT NOT NULL,
                        matched_text TEXT,
                        aspect_sentiment TEXT NOT NULL,
                        aspect_sentiment_score REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (review_id) REFERENCES reviews (review_id)
                    )
                ''')
                
                # Create tokens table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS tokens (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        review_id TEXT NOT NULL,
                        token_text TEXT NOT NULL,
                        token_type TEXT NOT NULL, -- 'positive' or 'negative'
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (review_id) REFERENCES reviews (review_id)
                    )
                ''')
                
                # Create summaries table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        statistical_summary TEXT, -- JSON
                        textual_summary TEXT,
                        aspect_summaries TEXT, -- JSON
                        created_at TEXT NOT NULL
                    )
                ''')
                
                # Create insights table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        pain_points TEXT, -- JSON
                        recommendations TEXT, -- JSON
                        trends TEXT, -- JSON
                        created_at TEXT NOT NULL
                    )
                ''')
                
                # Create indexes for better performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reviews_timestamp ON reviews(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reviews_language ON reviews(language)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_aspects_review_id ON aspects(review_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_aspects_label ON aspects(aspect_label)')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def save_processed_results(self, results: Dict[str, Any]) -> str:
        """
        Save complete processing results to database.
        
        Args:
            results: Complete results dictionary from pipeline
            
        Returns:
            Batch ID for the saved results
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with self._lock:
                with self._get_connection() as conn:
                    current_time = datetime.now().isoformat()
                    
                    # Save reviews
                    reviews = results.get("reviews", [])
                    for review in reviews:
                        self._save_review(conn, review, current_time)
                    
                    # Save summary
                    summary = results.get("summary", {})
                    if summary:
                        self._save_summary(conn, batch_id, summary, current_time)
                    
                    # Save insights
                    insights = results.get("insights", {})
                    if insights:
                        self._save_insights(conn, batch_id, insights, current_time)
                    
                    conn.commit()
                    logger.info(f"Saved processing results with batch_id: {batch_id}")
                    
            return batch_id
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _save_review(self, conn: sqlite3.Connection, review: Dict[str, Any], current_time: str):
        """Save individual review to database."""
        try:
            review_id = str(review.get("review_id", ""))
            
            # Insert or update review
            conn.execute('''
                INSERT OR REPLACE INTO reviews (
                    review_id, original_text, translated_text, language, timestamp,
                    sentiment, sentiment_score, intent, intent_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                review_id,
                review.get("original_text", ""),
                review.get("translated_text"),
                review.get("language", "Unknown"),
                review.get("timestamp", current_time),
                review.get("sentiment", "Neutral"),
                review.get("sentiment_score", 0.5),
                review.get("intent", "General Feedback"),
                review.get("intent_score", 0.5),
                current_time,
                current_time
            ))
            
            # Clear existing aspects and tokens for this review
            conn.execute('DELETE FROM aspects WHERE review_id = ?', (review_id,))
            conn.execute('DELETE FROM tokens WHERE review_id = ?', (review_id,))
            
            # Save aspects
            aspects = review.get("aspects", [])
            for aspect in aspects:
                conn.execute('''
                    INSERT INTO aspects (
                        review_id, aspect_id, aspect_label, matched_text,
                        aspect_sentiment, aspect_sentiment_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    review_id,
                    aspect.get("aspect_id"),
                    aspect.get("aspect_label", ""),
                    aspect.get("matched_text", ""),
                    aspect.get("aspect_sentiment", "Neutral"),
                    aspect.get("aspect_sentiment_score", 0.5),
                    current_time
                ))
            
            # Save tokens
            positive_tokens = review.get("tokens_positive", [])
            negative_tokens = review.get("tokens_negative", [])
            
            for token in positive_tokens:
                conn.execute('''
                    INSERT INTO tokens (review_id, token_text, token_type, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (review_id, token, "positive", current_time))
            
            for token in negative_tokens:
                conn.execute('''
                    INSERT INTO tokens (review_id, token_text, token_type, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (review_id, token, "negative", current_time))
                
        except Exception as e:
            logger.error(f"Error saving review {review.get('review_id')}: {str(e)}")
            raise
    
    def _save_summary(self, conn: sqlite3.Connection, batch_id: str, summary: Dict[str, Any], current_time: str):
        """Save summary data to database."""
        try:
            conn.execute('''
                INSERT INTO summaries (
                    batch_id, statistical_summary, textual_summary, aspect_summaries, created_at
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                batch_id,
                json.dumps(summary.get("statistical_summary", {})),
                summary.get("textual_summary", ""),
                json.dumps(summary.get("aspect_summaries", {})),
                current_time
            ))
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            raise
    
    def _save_insights(self, conn: sqlite3.Connection, batch_id: str, insights: Dict[str, Any], current_time: str):
        """Save insights data to database."""
        try:
            conn.execute('''
                INSERT INTO insights (
                    batch_id, pain_points, recommendations, trends, created_at
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                batch_id,
                json.dumps(insights.get("pain_points", [])),
                json.dumps(insights.get("recommendations", [])),
                json.dumps(insights.get("trends", {})),
                current_time
            ))
        except Exception as e:
            logger.error(f"Error saving insights: {str(e)}")
            raise
    
    def get_review_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Get review by ID with associated data."""
        try:
            with self._get_connection() as conn:
                # Get review data
                review_row = conn.execute(
                    'SELECT * FROM reviews WHERE review_id = ?', (review_id,)
                ).fetchone()
                
                if not review_row:
                    return None
                
                # Convert to dictionary
                review = dict(review_row)
                
                # Get aspects
                aspect_rows = conn.execute(
                    'SELECT * FROM aspects WHERE review_id = ?', (review_id,)
                ).fetchall()
                
                review["aspects"] = [dict(row) for row in aspect_rows]
                
                # Get tokens
                positive_tokens = conn.execute(
                    'SELECT token_text FROM tokens WHERE review_id = ? AND token_type = ?',
                    (review_id, "positive")
                ).fetchall()
                
                negative_tokens = conn.execute(
                    'SELECT token_text FROM tokens WHERE review_id = ? AND token_type = ?',
                    (review_id, "negative")
                ).fetchall()
                
                review["tokens_positive"] = [row[0] for row in positive_tokens]
                review["tokens_negative"] = [row[0] for row in negative_tokens]
                
                return review
                
        except Exception as e:
            logger.error(f"Error getting review {review_id}: {str(e)}")
            return None
    
    def get_all_reviews(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all reviews with associated data."""
        try:
            with self._get_connection() as conn:
                query = 'SELECT * FROM reviews ORDER BY created_at DESC'
                if limit:
                    query += f' LIMIT {limit}'
                
                review_rows = conn.execute(query).fetchall()
                
                reviews = []
                for row in review_rows:
                    review = dict(row)
                    
                    # Get aspects
                    aspect_rows = conn.execute(
                        'SELECT * FROM aspects WHERE review_id = ?', (review["review_id"],)
                    ).fetchall()
                    
                    review["aspects"] = [dict(aspect_row) for aspect_row in aspect_rows]
                    
                    # Get tokens
                    positive_tokens = conn.execute(
                        'SELECT token_text FROM tokens WHERE review_id = ? AND token_type = ?',
                        (review["review_id"], "positive")
                    ).fetchall()
                    
                    negative_tokens = conn.execute(
                        'SELECT token_text FROM tokens WHERE review_id = ? AND token_type = ?',
                        (review["review_id"], "negative")
                    ).fetchall()
                    
                    review["tokens_positive"] = [row[0] for row in positive_tokens]
                    review["tokens_negative"] = [row[0] for row in negative_tokens]
                    
                    reviews.append(review)
                
                return reviews
                
        except Exception as e:
            logger.error(f"Error getting all reviews: {str(e)}")
            return []
    
    def get_reviews_by_filter(self, 
                            sentiment: str = None, 
                            language: str = None, 
                            intent: str = None,
                            start_date: str = None,
                            end_date: str = None) -> List[Dict[str, Any]]:
        """Get reviews filtered by criteria."""
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM reviews WHERE 1=1"
                params = []
                
                if sentiment:
                    query += " AND sentiment = ?"
                    params.append(sentiment)
                
                if language:
                    query += " AND language = ?"
                    params.append(language)
                
                if intent:
                    query += " AND intent = ?"
                    params.append(intent)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                
                query += " ORDER BY created_at DESC"
                
                review_rows = conn.execute(query, params).fetchall()
                
                # For efficiency, return basic review data without aspects/tokens for filtering
                reviews = [dict(row) for row in review_rows]
                return reviews
                
        except Exception as e:
            logger.error(f"Error getting filtered reviews: {str(e)}")
            return []
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from database."""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Total reviews
                total_reviews = conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
                stats['total_reviews'] = total_reviews
                
                # Sentiment distribution
                sentiment_dist = conn.execute('''
                    SELECT sentiment, COUNT(*) as count 
                    FROM reviews 
                    GROUP BY sentiment
                ''').fetchall()
                
                stats['sentiment_distribution'] = {row[0]: row[1] for row in sentiment_dist}
                
                # Language distribution
                lang_dist = conn.execute('''
                    SELECT language, COUNT(*) as count 
                    FROM reviews 
                    GROUP BY language
                ''').fetchall()
                
                stats['language_distribution'] = {row[0]: row[1] for row in lang_dist}
                
                # Intent distribution
                intent_dist = conn.execute('''
                    SELECT intent, COUNT(*) as count 
                    FROM reviews 
                    GROUP BY intent
                ''').fetchall()
                
                stats['intent_distribution'] = {row[0]: row[1] for row in intent_dist}
                
                # Top aspects
                top_aspects = conn.execute('''
                    SELECT aspect_label, COUNT(*) as count 
                    FROM aspects 
                    GROUP BY aspect_label 
                    ORDER BY count DESC 
                    LIMIT 10
                ''').fetchall()
                
                stats['top_aspects'] = {row[0]: row[1] for row in top_aspects}
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting summary statistics: {str(e)}")
            return {}
    
    def clear_all_data(self):
        """Clear all data from database (use with caution)."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute('DELETE FROM tokens')
                    conn.execute('DELETE FROM aspects')
                    conn.execute('DELETE FROM insights')
                    conn.execute('DELETE FROM summaries')
                    conn.execute('DELETE FROM reviews')
                    conn.commit()
                    
            logger.info("All database data cleared")
            
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            raise
    
    def export_to_csv(self, output_file: str) -> bool:
        """Export reviews to CSV file."""
        try:
            import csv
            
            reviews = self.get_all_reviews()
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                if not reviews:
                    return True
                
                # Define fieldnames
                fieldnames = [
                    'review_id', 'original_text', 'translated_text', 'language',
                    'timestamp', 'sentiment', 'sentiment_score', 'intent', 'intent_score',
                    'aspects', 'tokens_positive', 'tokens_negative'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for review in reviews:
                    # Convert complex fields to strings
                    review_copy = review.copy()
                    review_copy['aspects'] = json.dumps(review['aspects'])
                    review_copy['tokens_positive'] = json.dumps(review['tokens_positive'])
                    review_copy['tokens_negative'] = json.dumps(review['tokens_negative'])
                    
                    writer.writerow(review_copy)
            
            logger.info(f"Exported {len(reviews)} reviews to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        try:
            with self._get_connection() as conn:
                # Get table sizes
                tables = ['reviews', 'aspects', 'tokens', 'summaries', 'insights']
                table_sizes = {}
                
                for table in tables:
                    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                    table_sizes[table] = count
                
                # Get database file size
                import os
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'database_path': self.db_path,
                    'database_size_bytes': db_size,
                    'table_sizes': table_sizes,
                    'total_records': sum(table_sizes.values())
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
            return {'error': str(e)}
