# ABOUTME: Geographic utility functions for distance and bounding box calculations
# ABOUTME: Used by USGS and other reference station search functionality

from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.

    Args:
        lat1 (float): Latitude of point 1 in decimal degrees
        lon1 (float): Longitude of point 1 in decimal degrees
        lat2 (float): Latitude of point 2 in decimal degrees
        lon2 (float): Longitude of point 2 in decimal degrees

    Returns:
        float: Distance between the points in kilometers
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_km = 6371.0 * c

    return distance_km


def get_bounding_box(latitude_deg, longitude_deg, radius_km):
    """
    Calculate a bounding box around a center point with specified radius.

    Args:
        latitude_deg (float): Latitude of center point in decimal degrees
        longitude_deg (float): Longitude of center point in decimal degrees
        radius_km (float): Radius in kilometers

    Returns:
        tuple: (min_lat, max_lat, min_lon, max_lon)
    """
    lat_degrees_per_km = 1 / 110.574
    lon_degrees_per_km = 1 / (111.320 * cos(radians(latitude_deg)))

    lat_change = lat_degrees_per_km * radius_km
    lon_change = lon_degrees_per_km * radius_km

    min_lat = latitude_deg - lat_change
    max_lat = latitude_deg + lat_change
    min_lon = longitude_deg - lon_change
    max_lon = longitude_deg + lon_change

    return min_lat, max_lat, min_lon, max_lon
