# ABOUTME: Cache management for dashboard performance optimization
# ABOUTME: Provides file-based and Streamlit session caching utilities

import streamlit as st
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import hashlib
from functools import wraps
import time

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache" / "dashboard"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration times
CACHE_EXPIRY = {
    "external_api": timedelta(days=7),  # External API data cached for 1 week
    "processed_data": timedelta(days=30),  # Processed data cached for 1 month
    "aggregations": timedelta(hours=24),  # Daily aggregations cached for 24 hours
    "plots": timedelta(hours=12),  # Plot data cached for 12 hours
}


def get_cache_key(func_name, *args, **kwargs):
    """Generate a unique cache key based on function name and arguments."""
    # Create a string representation of arguments
    key_parts = [func_name]
    key_parts.extend(str(arg) for arg in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "_".join(key_parts)

    # Generate hash for filename safety
    return hashlib.md5(key_string.encode()).hexdigest()


def is_cache_valid(cache_file, expiry_delta):
    """Check if a cache file exists and is still valid."""
    if not cache_file.exists():
        return False

    # Check age
    file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
    return file_age < expiry_delta


def disk_cache(cache_type="processed_data"):
    """Decorator for disk-based caching of expensive operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            cache_file = CACHE_DIR / f"{cache_key}.pkl"

            # Check if valid cache exists
            if is_cache_valid(cache_file, CACHE_EXPIRY[cache_type]):
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    # Cache file may be corrupted; regenerate silently
                    print(f"Cache load failed for {func.__name__}, regenerating: {e}")

            # Execute function and cache result
            result = func(*args, **kwargs)

            # Save to cache
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception as e:
                st.warning(f"Failed to cache {func.__name__}: {str(e)}")

            return result

        return wrapper

    return decorator


@st.cache_data(ttl=3600)  # Streamlit cache for 1 hour
def load_and_aggregate_subhourly_data(file_path, aggregation="1H"):
    """
    Load sub-hourly data and create time-based aggregations.

    This function efficiently handles large sub-hourly datasets by:
    1. Loading data in chunks if needed
    2. Creating hourly/daily aggregations
    3. Caching results
    """
    # Check if file exists
    if not Path(file_path).exists():
        return None

    # For very large files, read in chunks
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

    if file_size_mb > 100:  # If larger than 100MB
        # Read in chunks and aggregate
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=10000, parse_dates=["datetime"]):
            # Aggregate chunk
            chunk_agg = (
                chunk.set_index("datetime")
                .resample(aggregation)
                .agg(
                    {
                        "rh": ["mean", "std", "count", "min", "max"],
                        "amplitude": "mean",
                        "rh_dot": "mean" if "rh_dot" in chunk.columns else None,
                    }
                )
            )
            chunks.append(chunk_agg)

        # Combine chunks
        data = pd.concat(chunks)
    else:
        # Read entire file
        data = pd.read_csv(file_path, parse_dates=["datetime"])

        # Create aggregations
        data = (
            data.set_index("datetime")
            .resample(aggregation)
            .agg(
                {
                    "rh": ["mean", "std", "count", "min", "max"],
                    "amplitude": "mean",
                    "rh_dot": "mean" if "rh_dot" in data.columns else None,
                }
            )
        )

    # Flatten column names
    data.columns = ["_".join(col).strip() for col in data.columns.values]

    return data.reset_index()


@disk_cache("external_api")
def fetch_external_data_cached(data_source, station_id, year, date_range=None):
    """
    Cached wrapper for external API calls.

    This function caches external API responses to disk to avoid
    repeated API calls for the same data.
    """
    from dashboard_components.data_loader import fetch_coops_data

    if data_source == "coops":
        return fetch_coops_data(station_id, year, doy_range=date_range)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def create_data_summary(df, columns_to_summarize):
    """
    Create a summary of data for quick overview without loading full dataset.
    """
    summary = {
        "row_count": len(df),
        "date_range": {
            "start": df["date"].min().isoformat() if "date" in df.columns else None,
            "end": df["date"].max().isoformat() if "date" in df.columns else None,
        },
        "columns": df.columns.tolist(),
        "statistics": {},
    }

    for col in columns_to_summarize:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            summary["statistics"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "non_null_count": int(df[col].notna().sum()),
            }

    return summary


class DataPreloader:
    """
    Manages background data preloading for improved responsiveness.
    """

    def __init__(self):
        self.preload_status = {}

    def preload_station_data(self, station_id, year):
        """
        Preload commonly used data for a station in the background.
        """
        import threading

        def _preload():
            # Load main data files
            from dashboard_components.data_loader import load_station_data

            load_station_data(station_id, year)

            # Cache external data if not already cached
            fetch_external_data_cached("coops", station_id, year)

            self.preload_status[f"{station_id}_{year}"] = True

        # Start preloading in background thread
        thread = threading.Thread(target=_preload)
        thread.daemon = True
        thread.start()

    def is_preloaded(self, station_id, year):
        """Check if data for a station/year is preloaded."""
        return self.preload_status.get(f"{station_id}_{year}", False)


# Singleton instance
_preloader = DataPreloader()


def get_preloader():
    """Get the singleton DataPreloader instance."""
    return _preloader


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        if execution_time > 5:  # Log slow operations
            st.warning(f"{func.__name__} took {execution_time:.1f}s to execute")

        return result

    return wrapper


# Export all utilities
__all__ = [
    "disk_cache",
    "load_and_aggregate_subhourly_data",
    "fetch_external_data_cached",
    "create_data_summary",
    "DataPreloader",
    "get_preloader",
    "monitor_performance",
    "CACHE_DIR",
]
