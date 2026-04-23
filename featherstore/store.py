"""Core FeatherStore class for managing feature groups using DuckDB and Parquet."""

import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class FeatherStore:
    """Lightweight local feature store backed by DuckDB and Parquet files."""

    def __init__(self, store_path: str = ".featherstore"):
        """
        Initialize the feature store.

        Args:
            store_path: Directory where feature groups (Parquet files) are stored.
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(database=str(self.store_path / "catalog.duckdb"))
        self._init_catalog()

    def _init_catalog(self) -> None:
        """Create the feature group catalog table if it doesn't exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_groups (
                name        VARCHAR PRIMARY KEY,
                path        VARCHAR NOT NULL,
                created_at  TIMESTAMP DEFAULT current_timestamp,
                description VARCHAR
            )
            """
        )

    def save(self, name: str, df: pd.DataFrame, description: str = "") -> None:
        """
        Save a DataFrame as a named feature group.

        Args:
            name: Unique name for the feature group.
            df: DataFrame containing the features.
            description: Optional human-readable description.
        """
        parquet_path = self.store_path / f"{name}.parquet"
        df.to_parquet(parquet_path, index=False)

        self._conn.execute(
            """
            INSERT INTO feature_groups (name, path, description)
            VALUES (?, ?, ?)
            ON CONFLICT (name) DO UPDATE SET
                path = excluded.path,
                description = excluded.description,
                created_at = current_timestamp
            """,
            [name, str(parquet_path), description],
        )

    def load(self, name: str, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Load a feature group by name.

        Args:
            name: Name of the feature group to load.
            columns: Optional list of columns to select.

        Returns:
            DataFrame with the requested features.
        """
        row = self._conn.execute(
            "SELECT path FROM feature_groups WHERE name = ?", [name]
        ).fetchone()

        if row is None:
            raise KeyError(f"Feature group '{name}' not found in store.")

        parquet_path = row[0]
        col_expr = ", ".join(columns) if columns else "*"
        return self._conn.execute(
            f"SELECT {col_expr} FROM read_parquet('{parquet_path}')"
        ).df()

    def list_groups(self) -> pd.DataFrame:
        """Return a DataFrame listing all registered feature groups."""
        return self._conn.execute(
            "SELECT name, description, created_at FROM feature_groups ORDER BY name"
        ).df()

    def delete(self, name: str) -> None:
        """Remove a feature group from the store."""
        row = self._conn.execute(
            "SELECT path FROM feature_groups WHERE name = ?", [name]
        ).fetchone()
        if row is None:
            raise KeyError(f"Feature group '{name}' not found in store.")
        parquet_path = Path(row[0])
        if parquet_path.exists():
            parquet_path.unlink()
        self._conn.execute("DELETE FROM feature_groups WHERE name = ?", [name])

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()

    def __repr__(self) -> str:
        return f"FeatherStore(store_path='{self.store_path}')"
