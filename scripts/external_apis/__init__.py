"""
External APIs Module for GNSS-IR Processing

This module provides integration with external data sources to enhance
GNSS-IR analysis with oceanographic and meteorological context:

- NOAA CO-OPS API: Tide predictions and water level observations
- NDBC Buoy Data: Wave height, wind speed/direction, meteorological data
"""

from .noaa_coops import NOAACOOPSClient
from .ndbc_client import NDBCClient

__all__ = ['NOAACOOPSClient', 'NDBCClient']