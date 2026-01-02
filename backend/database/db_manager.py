"""
Database manager for LLM Arena Analytics.

This module provides database connection management and CRUD operations
for the analytics database.
"""

from typing import Dict, List, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import os


class DatabaseManager:
    """Manager for database connections and operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pool_size: int = 5
    ) -> None:
        """
        Initialize the database manager.

        Reads connection details from environment variables if not provided:
        - DB_HOST (default: localhost)
        - DB_PORT (default: 5432)
        - DB_NAME (default: llm_arena_analytics)
        - DB_USER (default: postgres)
        - DB_PASSWORD (required)

        Args:
            host: Database host (overrides DB_HOST env var)
            port: Database port (overrides DB_PORT env var)
            database: Database name (overrides DB_NAME env var)
            user: Database user (overrides DB_USER env var)
            password: Database password (overrides DB_PASSWORD env var)
            pool_size: Connection pool size
        """
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.database = database or os.getenv("DB_NAME", "llm_arena_analytics")
        self.user = user or os.getenv("DB_USER", "postgres")
        self.password = password or os.getenv("DB_PASSWORD", "")
        self.pool_size = pool_size
        self.pool: Optional[SimpleConnectionPool] = None

    def connect(self) -> None:
        """Create connection pool."""
        try:
            self.pool = SimpleConnectionPool(
                1,
                self.pool_size,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create connection pool: {e}")

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Yields:
            Database connection
        """
        if self.pool is None:
            self.connect()

        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of dictionaries containing query results
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]

    def execute_update(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount

    def insert_model(
        self,
        name: str,
        provider: Optional[str] = None,
        model_family: Optional[str] = None
    ) -> int:
        """
        Insert a new model into the database.

        Args:
            name: Model name
            provider: Model provider
            model_family: Model family

        Returns:
            ID of the inserted model
        """
        query = """
            INSERT INTO models (name, provider, model_family)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO UPDATE
            SET provider = EXCLUDED.provider,
                model_family = EXCLUDED.model_family,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """
        result = self.execute_query(query, (name, provider, model_family))
        return result[0]['id'] if result else None

    def insert_arena_score(
        self,
        model_id: int,
        score: float,
        rank: int,
        category: Optional[str] = None,
        date: Optional[str] = None
    ) -> int:
        """
        Insert an arena score/ranking for a model.

        Args:
            model_id: ID of the model
            score: Elo rating or score value
            rank: Rank position
            category: Optional category (e.g., 'overall', 'coding', 'math')
            date: Optional date string (YYYY-MM-DD), defaults to current date

        Returns:
            ID of the inserted record
        """
        query = """
            INSERT INTO arena_rankings (model_id, elo_rating, rank_position, category, recorded_at)
            VALUES (%s, %s, %s, %s, COALESCE(%s::TIMESTAMPTZ, CURRENT_TIMESTAMP))
            RETURNING id
        """
        result = self.execute_query(query, (model_id, score, rank, category, date))
        return result[0]['id'] if result else None

    def insert_pricing(
        self,
        model_id: int,
        input_price: float,
        output_price: float,
        date: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> int:
        """
        Insert pricing data for a model.

        Args:
            model_id: ID of the model
            input_price: Cost per input token
            output_price: Cost per output token
            date: Optional date string (YYYY-MM-DD), defaults to current date
            provider: Optional provider name (for reference)
            model_name: Optional model name (for reference)

        Returns:
            ID of the inserted record
        """
        query = """
            INSERT INTO pricing_data (model_id, input_cost_per_token, output_cost_per_token, 
                                     effective_date, provider, model_name)
            VALUES (%s, %s, %s, COALESCE(%s::DATE, CURRENT_DATE), %s, %s)
            ON CONFLICT (model_id, effective_date) DO UPDATE
            SET input_cost_per_token = EXCLUDED.input_cost_per_token,
                output_cost_per_token = EXCLUDED.output_cost_per_token,
                provider = EXCLUDED.provider,
                model_name = EXCLUDED.model_name
            RETURNING id
        """
        result = self.execute_query(query, (model_id, input_price, output_price, date, provider, model_name))
        return result[0]['id'] if result else None

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get all models from the database.

        Returns:
            List of model dictionaries
        """
        query = "SELECT * FROM models ORDER BY name"
        return self.execute_query(query)

    def get_arena_history(
        self,
        model_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get arena ranking history for a model over a specified time period.

        Args:
            model_id: ID of the model
            days: Number of days to look back (default: 30)

        Returns:
            List of arena ranking records ordered by date
        """
        query = """
            SELECT * FROM arena_rankings
            WHERE model_id = %s
            AND recorded_at >= CURRENT_TIMESTAMP - make_interval(days => %s)
            ORDER BY recorded_at DESC
        """
        return self.execute_query(query, (model_id, days))

    def get_latest_pricing(self) -> List[Dict[str, Any]]:
        """
        Get the latest pricing for all models.

        Returns:
            List of pricing records with the most recent effective_date for each model
        """
        query = """
            SELECT DISTINCT ON (model_id)
                pd.*,
                m.name as model_name,
                m.provider
            FROM pricing_data pd
            JOIN models m ON pd.model_id = m.id
            ORDER BY model_id, effective_date DESC
        """
        return self.execute_query(query)

    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a model by name.

        Args:
            name: Model name

        Returns:
            Model dictionary or None if not found
        """
        query = "SELECT * FROM models WHERE name = %s"
        results = self.execute_query(query, (name,))
        return results[0] if results else None

    def close(self) -> None:
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()

