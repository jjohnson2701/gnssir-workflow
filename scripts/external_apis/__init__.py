# ABOUTME: External API clients for oceanographic and meteorological data
# ABOUTME: Integrates NOAA CO-OPS tide data and NDBC buoy measurements

from .noaa_coops import NOAACOOPSClient
from .ndbc_client import NDBCClient

__all__ = ["NOAACOOPSClient", "NDBCClient"]
