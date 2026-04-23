# featherstore

Lightweight local feature store for rapid ML experimentation using DuckDB and Parquet.

---

## Installation

```bash
pip install featherstore
```

Or install from source:

```bash
git clone https://github.com/yourusername/featherstore.git
cd featherstore && pip install -e .
```

---

## Usage

```python
from featherstore import FeatherStore

# Initialize a local store
store = FeatherStore(path="./my_feature_store")

# Write features
store.write("user_features", df)

# Read features back
features = store.read("user_features", columns=["user_id", "age", "spend_30d"])

# Query with SQL via DuckDB
result = store.query("SELECT user_id, spend_30d FROM user_features WHERE spend_30d > 100")
```

Features are stored as Parquet files and queried in-process via DuckDB — no server required.

---

## Features

- 🪶 Zero-config local feature store
- ⚡ Fast reads powered by DuckDB + Parquet
- 🔍 Full SQL query support
- 📦 Versioned feature sets
- 🐍 Pandas & Polars compatible

---

## Requirements

- Python 3.8+
- `duckdb >= 0.9`
- `pandas >= 1.5`
- `pyarrow >= 12.0`

---

## License

This project is licensed under the [MIT License](LICENSE).