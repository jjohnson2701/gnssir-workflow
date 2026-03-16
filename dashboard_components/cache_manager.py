# ABOUTME: Cache management for dashboard performance optimization.
# ABOUTME: Provides file-based caching decorator and performance monitoring.

import streamlit as st
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


__all__ = [
    "disk_cache",
    "monitor_performance",
    "CACHE_DIR",
]
