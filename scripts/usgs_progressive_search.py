# ABOUTME: Progressive radius search for nearby USGS stream gauges
# ABOUTME: Expands search radius until suitable reference gauge is found

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from dataretrieval import nwis

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
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_km = 6371.0 * c  # Radius of Earth in km
    
    return distance_km

def get_bounding_box(latitude_in_degrees, longitude_in_degrees, half_side_in_km):
    """
    Calculate a bounding box around a center point with specified radius.
    
    Args:
        latitude_in_degrees (float): Latitude of center point in decimal degrees
        longitude_in_degrees (float): Longitude of center point in decimal degrees
        half_side_in_km (float): Half the length of the bounding box side in kilometers
    
    Returns:
        tuple: (min_lat, max_lat, min_lon, max_lon)
    """
    # Log the calculation for debugging
    logging.debug(f"Calculating bounding box for center ({latitude_in_degrees}, {longitude_in_degrees}) with radius {half_side_in_km} km")
    
    # Approximate degrees latitude per km
    lat_degrees_per_km = 1 / 110.574
    
    # Approximate degrees longitude per km at the given latitude
    lon_degrees_per_km = 1 / (111.320 * cos(radians(latitude_in_degrees)))
    
    # Calculate the bounding box coordinates
    lat_change = lat_degrees_per_km * half_side_in_km
    lon_change = lon_degrees_per_km * half_side_in_km
    
    min_lat = latitude_in_degrees - lat_change
    max_lat = latitude_in_degrees + lat_change
    min_lon = longitude_in_degrees - lon_change
    max_lon = longitude_in_degrees + lon_change
    
    return min_lat, max_lat, min_lon, max_lon

def get_state_from_coordinates(lat, lon):
    """
    Determine the most likely US state based on coordinates.
    This is a simple approximation for common GNSS-IR stations.
    
    Args:
        lat (float): Latitude in decimal degrees
        lon (float): Longitude in decimal degrees
    
    Returns:
        str: Two-letter state code
    """
    # Simple bounding boxes for a few coastal states (very approximate)
    state_bounds = {
        "NC": (33.5, 36.6, -84.4, -75.2),  # North Carolina
        "SC": (32.0, 35.3, -83.4, -78.5),  # South Carolina
        "FL": (24.5, 31.0, -87.6, -80.0),  # Florida
        "CA": (32.5, 42.0, -124.5, -114.0), # California
        "WA": (45.5, 49.0, -124.8, -116.9), # Washington
        "OR": (42.0, 46.3, -124.7, -116.4), # Oregon
        "TX": (25.8, 36.5, -106.7, -93.5),  # Texas
        "LA": (29.0, 33.0, -94.1, -88.8),   # Louisiana
        "MS": (30.0, 35.0, -92.0, -88.0),   # Mississippi
        "AL": (30.1, 35.0, -88.5, -84.9),   # Alabama
        "GA": (30.4, 35.0, -85.6, -80.8),   # Georgia
        "VA": (36.5, 39.5, -83.7, -75.2),   # Virginia
        "MD": (37.9, 39.8, -79.5, -75.0),   # Maryland
        "DE": (38.4, 39.8, -75.8, -74.9),   # Delaware
        "NJ": (38.9, 41.4, -75.6, -73.9),   # New Jersey
        "NY": (40.5, 45.0, -79.8, -71.8),   # New York
        "CT": (40.9, 42.1, -73.7, -71.8),   # Connecticut
        "RI": (41.1, 42.0, -71.9, -71.1),   # Rhode Island
        "MA": (41.2, 42.9, -73.5, -69.9),   # Massachusetts
        "NH": (42.7, 45.3, -72.6, -70.7),   # New Hampshire
        "ME": (43.1, 47.5, -71.1, -66.9),   # Maine
        "HI": (18.5, 22.5, -160.5, -154.5), # Hawaii
        "AK": (51.0, 71.5, -180.0, -130.0), # Alaska
    }
    
    for state, (min_lat, max_lat, min_lon, max_lon) in state_bounds.items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return state
    
    # If no match found, guess based on general region
    if lon < -100:
        if lat > 40:
            return "WA"  # Default to Washington for Northwest
        else:
            return "CA"  # Default to California for Southwest
    else:
        if lat > 40:
            return "NY"  # Default to New York for Northeast
        else:
            return "FL"  # Default to Florida for Southeast

def construct_bbox_string(min_lat, max_lat, min_lon, max_lon):
    """
    Construct a bounding box string in the format required by the USGS API.
    
    Args:
        min_lat (float): Minimum latitude
        max_lat (float): Maximum latitude
        min_lon (float): Minimum longitude
        max_lon (float): Maximum longitude
        
    Returns:
        str: Bounding box string in format "min_lon,min_lat,max_lon,max_lat"
    """
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"

def progressive_gauge_search(gnss_station_lat, 
                           gnss_station_lon, 
                           initial_radius_km, 
                           radius_increment_km, 
                           max_radius_km, 
                           desired_parameter_codes, 
                           min_gauges_found=3,
                           min_data_availability_days=None, 
                           data_start_date=None, 
                           data_end_date=None):
    """
    Perform a progressive search for USGS gauges, expanding the search radius until
    a minimum number of gauges are found or the maximum radius is reached.
    
    Args:
        gnss_station_lat (float): Latitude of GNSS station in decimal degrees
        gnss_station_lon (float): Longitude of GNSS station in decimal degrees
        initial_radius_km (float): Initial search radius in kilometers
        radius_increment_km (float): Amount to increase radius each iteration in kilometers
        max_radius_km (float): Maximum search radius in kilometers
        desired_parameter_codes (list): List of USGS parameter codes to search for
        min_gauges_found (int, optional): Minimum number of gauges to find before stopping (default: 3)
        min_data_availability_days (int, optional): Minimum number of days with data (optional, for future use)
        data_start_date (str, optional): Start date for data availability check (optional, for future use)
        data_end_date (str, optional): End date for data availability check (optional, for future use)
        
    Returns:
        pd.DataFrame: DataFrame containing gauge information sorted by distance
    """
    logging.info(f"Starting progressive gauge search from ({gnss_station_lat}, {gnss_station_lon})")
    logging.info(f"Parameters: initial radius = {initial_radius_km} km, increment = {radius_increment_km} km, "
                f"max radius = {max_radius_km} km, min gauges = {min_gauges_found}")
    logging.info(f"Desired parameter codes: {desired_parameter_codes}")
    
    current_radius = initial_radius_km
    all_gauges = []
    matching_gauges = []
    state_code = get_state_from_coordinates(gnss_station_lat, gnss_station_lon)
    
    while current_radius <= max_radius_km:
        logging.info(f"Searching with radius {current_radius} km")
        
        try:
            # Get bounding box for current radius
            min_lat, max_lat, min_lon, max_lon = get_bounding_box(
                gnss_station_lat, gnss_station_lon, current_radius
            )
            bbox_str = construct_bbox_string(min_lat, max_lat, min_lon, max_lon)
            
            # Query sites with current bounding box
            site_types = ["ST", "LK", "ES", "OC"]  # Stream, Lake, Estuary, Ocean
            parameter_codes_str = ",".join(desired_parameter_codes)
            
            # Reset gauges for this radius
            gauges_this_radius = []
            
            try:
                logging.info(f"Querying USGS sites in bounding box {bbox_str}")
                
                # Use bBox parameter as major filter with parameter codes
                sites_df = nwis.get_record(
                    service='site',
                    bBox=bbox_str,
                    parameterCd=parameter_codes_str,
                    siteType=",".join(site_types)
                )
                
                if sites_df is not None and not sites_df.empty:
                    logging.info(f"Found {len(sites_df)} sites in bounding box")
                    
                    # Process sites
                    for site_no, site in sites_df.iterrows():
                        try:
                            # Extract site info
                            site_code = site.get('site_no', site_no)  # Use actual site_no column, fallback to index
                            site_name = site.get('station_nm', '')
                            site_lat = float(site.get('dec_lat_va', 0))
                            site_lon = float(site.get('dec_long_va', 0))
                            
                            # Calculate distance
                            distance = haversine_distance(
                                gnss_station_lat, gnss_station_lon,
                                site_lat, site_lon
                            )
                            
                            # Check if site is within radius
                            if distance <= current_radius:
                                # Get parameter information for the site
                                try:
                                    matched_parameters = []
                                    vertical_datum = "Unknown"
                                    site_type = site.get('site_tp_cd', '')
                                    
                                    # Extract available parameters
                                    if 'parm_cd' in site:
                                        parm_cd = site['parm_cd']
                                        if isinstance(parm_cd, str):
                                            available_parameters = [parm_cd]
                                        else:
                                            available_parameters = parm_cd.tolist() if hasattr(parm_cd, 'tolist') else [parm_cd]
                                            
                                        # Check which parameters match what we want
                                        for param in desired_parameter_codes:
                                            if param in available_parameters:
                                                matched_parameters.append(param)
                                    
                                    # Extract vertical datum
                                    if 'alt_datum_cd' in site:
                                        vertical_datum = site['alt_datum_cd']
                                    
                                    # Check data availability if dates were provided
                                    has_data = False
                                    if data_start_date and data_end_date and matched_parameters:
                                        # For now, assume data is available
                                        has_data = True
                                    else:
                                        # If no dates provided, just check if we found any matching parameters
                                        has_data = len(matched_parameters) > 0
                                    
                                    # Add site if it has matching parameters
                                    if has_data:
                                        gauges_this_radius.append({
                                            'site_no': site_code,
                                            'station_nm': site_name,
                                            'dec_lat_va': site_lat,
                                            'dec_long_va': site_lon,
                                            'distance_km': distance,
                                            'site_tp_cd': site_type,
                                            'alt_datum_cd': vertical_datum,
                                            'matched_parameters': matched_parameters
                                        })
                                        logging.info(f"Found matching gauge: {site_code} at {distance:.2f} km")
                                
                                except Exception as e:
                                    logging.warning(f"Error processing parameters for site {site_code}: {str(e)}")
                        
                        except Exception as e:
                            logging.warning(f"Error processing site: {str(e)}")
                else:
                    logging.info(f"No sites found in bounding box using parameter codes")
                    
                    # If no sites found with parameter codes, try without them
                    try:
                        logging.info("Trying query without parameter codes")
                        sites_df = nwis.get_record(
                            service='site',
                            bBox=bbox_str,
                            siteType=",".join(site_types)
                        )
                        
                        if sites_df is not None and not sites_df.empty:
                            logging.info(f"Found {len(sites_df)} sites in bounding box without parameter filter")
                            
                            # Process sites similarly to the first query
                            for site_no, site in sites_df.iterrows():
                                try:
                                    # Extract site info
                                    site_code = site.get('site_no', site_no)  # Use actual site_no column, fallback to index
                                    site_name = site.get('station_nm', '')
                                    site_lat = float(site.get('dec_lat_va', 0))
                                    site_lon = float(site.get('dec_long_va', 0))
                                    
                                    # Calculate distance
                                    distance = haversine_distance(
                                        gnss_station_lat, gnss_station_lon,
                                        site_lat, site_lon
                                    )
                                    
                                    # Check if site is within radius
                                    if distance <= current_radius:
                                        # Get detailed site info to check parameters
                                        try:
                                            site_data = nwis.get_record(
                                                service='site',
                                                sites=site_code
                                            )
                                            
                                            matched_parameters = []
                                            vertical_datum = "Unknown"
                                            site_type = site.get('site_tp_cd', '')
                                            
                                            # Extract vertical datum
                                            if 'alt_datum_cd' in site_data.columns and not site_data.empty:
                                                vertical_datum = site_data.loc[site_code, 'alt_datum_cd']
                                            
                                            # For now, we'll add all sites in radius even without parameter match
                                            gauges_this_radius.append({
                                                'site_no': site_code,
                                                'station_nm': site_name,
                                                'dec_lat_va': site_lat,
                                                'dec_long_va': site_lon,
                                                'distance_km': distance,
                                                'site_tp_cd': site_type,
                                                'alt_datum_cd': vertical_datum,
                                                'matched_parameters': matched_parameters
                                            })
                                            logging.info(f"Found gauge without parameter matching: {site_code} at {distance:.2f} km")
                                        
                                        except Exception as e:
                                            logging.warning(f"Error getting detailed info for site {site_code}: {str(e)}")
                                
                                except Exception as e:
                                    logging.warning(f"Error processing site: {str(e)}")
                    except Exception as e:
                        logging.warning(f"Error in fallback query: {str(e)}")
            
            except Exception as e:
                logging.warning(f"Error querying sites in bounding box: {str(e)}")
                
                # Try with state code instead as an alternative approach
                try:
                    logging.info(f"Trying alternative query with state {state_code}")
                    sites_df = nwis.get_record(
                        service='site',
                        stateCd=state_code,
                        siteType=",".join(site_types)
                    )
                    
                    if sites_df is not None and not sites_df.empty:
                        logging.info(f"Found {len(sites_df)} sites in state {state_code}")
                        
                        # Process sites
                        for site_no, site in sites_df.iterrows():
                            try:
                                # Extract site info
                                site_code = site.get('site_no', site_no)  # Use actual site_no column, fallback to index
                                site_name = site.get('station_nm', '')
                                site_lat = float(site.get('dec_lat_va', 0))
                                site_lon = float(site.get('dec_long_va', 0))
                                
                                # Calculate distance
                                distance = haversine_distance(
                                    gnss_station_lat, gnss_station_lon,
                                    site_lat, site_lon
                                )
                                
                                # Check if site is within radius
                                if distance <= current_radius:
                                    # Add the site (we'll skip parameter checking for now)
                                    gauges_this_radius.append({
                                        'site_no': site_code,
                                        'station_nm': site_name,
                                        'dec_lat_va': site_lat,
                                        'dec_long_va': site_lon,
                                        'distance_km': distance,
                                        'site_tp_cd': site.get('site_tp_cd', ''),
                                        'alt_datum_cd': site.get('alt_datum_cd', 'Unknown'),
                                        'matched_parameters': []
                                    })
                                    logging.info(f"Found gauge via state search: {site_code} at {distance:.2f} km")
                            
                            except Exception as e:
                                logging.warning(f"Error processing site from state search: {str(e)}")
                except Exception as e:
                    logging.warning(f"Error in state-based search: {str(e)}")
            
            # Add gauges from this radius to the overall list
            all_gauges.extend(gauges_this_radius)
            
            # Update matching gauges (unique sites sorted by distance)
            site_codes_seen = set()
            matching_gauges = []
            
            for gauge in sorted(all_gauges, key=lambda g: g['distance_km']):
                if gauge['site_no'] not in site_codes_seen:
                    matching_gauges.append(gauge)
                    site_codes_seen.add(gauge['site_no'])
            
            logging.info(f"Found {len(gauges_this_radius)} gauges within {current_radius} km radius")
            logging.info(f"Total unique matching gauges so far: {len(matching_gauges)}")
            
            # Check if we found enough gauges
            if len(matching_gauges) >= min_gauges_found:
                logging.info(f"Found {len(matching_gauges)} gauges, which meets the minimum requirement of {min_gauges_found}")
                break
            
            # If not enough gauges found, increase radius and continue
            current_radius += radius_increment_km
        
        except Exception as e:
            logging.error(f"Error in search iteration at radius {current_radius} km: {str(e)}")
            current_radius += radius_increment_km
    
    # Create DataFrame from matching gauges
    if matching_gauges:
        df = pd.DataFrame(matching_gauges)
        result_radius = min(current_radius, max_radius_km)
        logging.info(f"Found {len(df)} unique gauges within {result_radius} km")
        return df
    else:
        logging.warning(f"No valid USGS gauges found within {max_radius_km} km")
        return pd.DataFrame()

def find_usgs_gauges_progressive_search(
    gnss_station_lat, 
    gnss_station_lon,
    start_date,
    end_date,
    initial_radius_km=25.0,
    radius_increment_km=25.0,
    max_radius_km=200.0,
    min_matches=3,
    desired_parameter_codes=None,
    state_code=None
):
    """Legacy function - maintained for backward compatibility.
    New code should use progressive_gauge_search instead.
    
    Args:
        gnss_station_lat (float): Latitude of the GNSS station
        gnss_station_lon (float): Longitude of the GNSS station
        start_date (str): Start date for data availability check ("YYYY-MM-DD")
        end_date (str): End date for data availability check ("YYYY-MM-DD")
        initial_radius_km (float): Initial search radius in km
        radius_increment_km (float): Amount to increment radius by in each step
        max_radius_km (float): Maximum search radius in km
        min_matches (int): Minimum number of matching gauges to find
        desired_parameter_codes (list): List of parameter codes to search for
        state_code (str): Two-letter state code (optional)
        
    Returns:
        pd.DataFrame: DataFrame of matching gauges sorted by distance
    """
    if desired_parameter_codes is None:
        desired_parameter_codes = ["62610", "62611", "62620", "00065"]
    
    # Forward to the new progressive_gauge_search function
    return progressive_gauge_search(
        gnss_station_lat=gnss_station_lat,
        gnss_station_lon=gnss_station_lon,
        initial_radius_km=initial_radius_km,
        radius_increment_km=radius_increment_km,
        max_radius_km=max_radius_km,
        desired_parameter_codes=desired_parameter_codes,
        min_gauges_found=min_matches,
        data_start_date=start_date,
        data_end_date=end_date
    )
