# ABOUTME: USGS gauge discovery and water level data retrieval
# ABOUTME: Uses dataretrieval.nwis to fetch instantaneous gage height values

import logging
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from dataretrieval import nwis
from datetime import datetime, timedelta

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
        str: Bounding box string in format "lon_sw,lat_sw,lon_ne,lat_ne"
    """
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
    
    # Format as "lon_sw,lat_sw,lon_ne,lat_ne"
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
    return bbox

def find_nearby_usgs_gauges(gnss_station_lat, gnss_station_lon, radius_km=50.0, 
                           desired_parameter_codes=None, state_code=None, huc=None):
    """Find nearby USGS gauges within a specified radius of a GNSS station.
    
    This function has been updated to use dataretrieval.nwis instead of pynwis.
    
    Args:
        gnss_station_lat (float): Latitude of the GNSS station in decimal degrees
        gnss_station_lon (float): Longitude of the GNSS station in decimal degrees
        radius_km (float, optional): Search radius in kilometers. Defaults to 50.0.
        desired_parameter_codes (list, optional): List of USGS parameter codes to prioritize. 
                                                Defaults to None.
        state_code (str, optional): Two-letter state code to narrow search. Defaults to None.
        huc (str, optional): Hydrologic Unit Code to narrow search. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame of nearby USGS gauges sorted by distance
    """
    logging.info(f"Finding nearby USGS gauges for ({gnss_station_lat:.6f}, {gnss_station_lon:.6f}) with radius {radius_km} km")
    
    # Import the progressive search module
    from scripts.usgs_progressive_search import progressive_gauge_search
    
    # Set default parameter codes if not provided
    if desired_parameter_codes is None:
        desired_parameter_codes = ["62610", "62611", "62620", "00065"]  # NAVD88, NGVD29, MSL, Gage Height
    
    # Default dates for a one year period (not critically important as we're just looking for gauges)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Call the progressive search with a single radius
    gauges_df = progressive_gauge_search(
        gnss_station_lat=gnss_station_lat,
        gnss_station_lon=gnss_station_lon,
        initial_radius_km=radius_km,
        radius_increment_km=radius_km,  # Not used as we'll stop after one iteration
        max_radius_km=radius_km,
        desired_parameter_codes=desired_parameter_codes,
        min_gauges_found=1,  # Just find whatever is available within the radius
        data_start_date=start_date,
        data_end_date=end_date
    )
    
    # Rename columns to match the expected format by other functions
    if gauges_df is not None and not gauges_df.empty:
        # Map the column names from progressive_gauge_search to the expected format
        column_mapping = {
            'site_no': 'site_code',
            'dec_lat_va': 'latitude',
            'dec_long_va': 'longitude',
            'site_tp_cd': 'site_type',
            'alt_datum_cd': 'vertical_datum'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in gauges_df.columns:
                gauges_df[new_col] = gauges_df[old_col]
        
        logging.info(f"Found {len(gauges_df)} USGS gauges within {radius_km} km")
        return gauges_df
    else:
        logging.warning(f"No USGS gauges found within {radius_km} km")
        return pd.DataFrame()

def fetch_usgs_gauge_data(usgs_site_code, parameter_code=None, start_date_str=None, end_date_str=None, 
                     service="iv", priority_parameter_codes=None):
    """
    Fetch water level data for a specific USGS gauge using dataretrieval.nwis.
    Tries multiple parameter codes and service types if initial attempt fails.
    
    Args:
        usgs_site_code (str): The 8-digit USGS site code
        parameter_code (str, optional): The specific USGS parameter code to fetch
        start_date_str (str): Start date in "YYYY-MM-DD" format
        end_date_str (str): End date in "YYYY-MM-DD" format
        service (str, optional): "iv" for instantaneous values, "dv" for daily values. 
                               Defaults to "iv".
        priority_parameter_codes (list, optional): List of parameter codes to try in order
                                               if parameter_code is not provided or fails
    
    Returns:
        pd.DataFrame: DataFrame containing the time series data
        dict: Metadata about the gauge and data
        str: The parameter code that was successfully used
    """
    logging.info(f"Fetching USGS gauge data for site {usgs_site_code}")
    logging.info(f"Time range: {start_date_str} to {end_date_str}")
    logging.debug(f"Original service requested: {service}")
    
    # If priority_parameter_codes not provided, use a default list or create from parameter_code
    if priority_parameter_codes is None:
        if parameter_code is not None:
            # Ensure parameter_code is the first in the list
            priority_parameter_codes = [parameter_code, "00065", "62610", "62611", "62620"]
        else:
            priority_parameter_codes = ["00065", "62610", "62611", "62620"]
    elif parameter_code is not None and parameter_code not in priority_parameter_codes:
        # Add the specified parameter_code to the front of the list if it's not already there
        priority_parameter_codes = [parameter_code] + priority_parameter_codes
    
    logging.info(f"Priority parameter codes to try: {priority_parameter_codes}")
    
    # Try each parameter code in order
    for param_code in priority_parameter_codes:
        logging.info(f"Trying parameter code {param_code}")
        
        # First try with 'iv' service for instantaneous values
        logging.info(f"Attempting to fetch 'iv' service data for site {usgs_site_code}, param {param_code}")
        
        try:
            # Log the exact function call for debugging
            logging.info(f"Querying NWIS with 'iv' service: site={usgs_site_code}, param={param_code}, start={start_date_str}, end={end_date_str}")
            
            # Fetch data using dataretrieval.nwis with iv service
            site_data_iv = nwis.get_record(
                sites=usgs_site_code,
                service="iv",
                parameterCd=param_code,
                start=start_date_str,
                end=end_date_str
            )
            
            # Check if data was returned for 'iv' service
            if site_data_iv is not None and not site_data_iv.empty:
                logging.info(f"✅ Successfully fetched 'iv' data for param {param_code}")
                logging.debug(f"Data found:\n{site_data_iv.head()}")
                
                # Create a simplified metadata dictionary
                metadata = {
                    'site_code': usgs_site_code,
                    'parameter_code': param_code,
                    'site_name': 'Unknown',
                    'parameter_desc': 'Unknown',
                    'units': 'Unknown',
                    'datum': 'Unknown',
                    'service': 'iv'
                }
                
                # Call function to enhance metadata
                metadata = enhance_gauge_metadata(usgs_site_code, param_code, site_data_iv, metadata)
                
                # Return the successful data, metadata, and parameter code
                return site_data_iv, metadata, param_code
            else:
                logging.warning(f"NWIS returned no data for site {usgs_site_code}, param {param_code}, service='iv'")
        except Exception as e:
            logging.error(f"Exception during NWIS 'iv' query for site {usgs_site_code}, param {param_code}: {e}")
        
        # Try the 'dv' (daily values) service for SAME parameter code before moving to next parameter
        logging.info(f"Attempting to fetch 'dv' service data for site {usgs_site_code}, param {param_code}")
        
        try:
            # Log the exact function call for debugging
            logging.info(f"Querying NWIS with 'dv' service: site={usgs_site_code}, param={param_code}, start={start_date_str}, end={end_date_str}, statCd='00003'")
            
            # Fetch data using dataretrieval.nwis with dv service and mean daily stats
            site_data_dv = nwis.get_record(
                sites=usgs_site_code,
                service="dv",
                parameterCd=param_code,
                start=start_date_str,
                end=end_date_str,
                statCd="00003"  # Mean daily value
            )
            
            # Check if data was returned for 'dv' service
            if site_data_dv is not None and not site_data_dv.empty:
                logging.info(f"✅ Successfully fetched 'dv' data for param {param_code}")
                logging.debug(f"Data found:\n{site_data_dv.head()}")
                
                # Create a simplified metadata dictionary
                metadata = {
                    'site_code': usgs_site_code,
                    'parameter_code': param_code,
                    'site_name': 'Unknown',
                    'parameter_desc': 'Unknown',
                    'units': 'Unknown',
                    'datum': 'Unknown',
                    'service': 'dv'
                }
                
                # Call function to enhance metadata
                metadata = enhance_gauge_metadata(usgs_site_code, param_code, site_data_dv, metadata)
                
                # Return the successful data, metadata, and parameter code
                return site_data_dv, metadata, param_code
            else:
                logging.warning(f"NWIS returned no data for site {usgs_site_code}, param {param_code}, service='dv'")
        except Exception as e:
            logging.error(f"Exception during NWIS 'dv' query for site {usgs_site_code}, param {param_code}: {e}")
    
    # If we've tried all parameter codes with both 'iv' and 'dv' services with no success
    logging.error(f"❌ No data found for site {usgs_site_code} with any parameter codes or services")
    return pd.DataFrame(), {}, None


def enhance_gauge_metadata(usgs_site_code, parameter_code, site_data, metadata):
    """
    Enhance the gauge metadata with site information.
    
    Args:
        usgs_site_code (str): The 8-digit USGS site code
        parameter_code (str): The parameter code used
        site_data (pd.DataFrame): The retrieved site data
        metadata (dict): Initial metadata dictionary
    
    Returns:
        dict: Enhanced metadata dictionary
    """
    # Get site information to populate metadata - try more forcefully this time
    try:
        # Debug the metadata fetching
        logging.info(f"Fetching detailed site metadata for {usgs_site_code} using nwis.get_record...")
        site_info = nwis.get_record(
            sites=usgs_site_code,
            service='site'
        )
        
        if site_info is not None and not site_info.empty:
            logging.info(f"Successfully retrieved site info for {usgs_site_code}")

            # Use iloc[0] since the index may be numeric, not the site code
            row = site_info.iloc[0]

            # Extract site name
            if 'station_nm' in site_info.columns:
                metadata['site_name'] = row['station_nm']
                logging.info(f"Site name: {metadata['site_name']}")

            # Extract vertical datum
            if 'alt_datum_cd' in site_info.columns:
                metadata['datum'] = row['alt_datum_cd']
                logging.info(f"Vertical datum: {metadata['datum']}")

            # Extract additional useful metadata and store in metadata dict
            for field in ['dec_lat_va', 'dec_long_va', 'alt_va', 'site_tp_cd', 'nat_aqfr_cd']:
                if field in site_info.columns:
                    value = row[field]
                    metadata[field] = value
                    logging.info(f"Site {field}: {value}")

            # Also store latitude/longitude with friendly names for easy access
            if 'dec_lat_va' in site_info.columns:
                metadata['latitude'] = row['dec_lat_va']
            if 'dec_long_va' in site_info.columns:
                metadata['longitude'] = row['dec_long_va']

            # Log all available columns for debugging
            logging.debug(f"Available site info columns: {site_info.columns.tolist()}")
        else:
            logging.warning(f"No site info found for site {usgs_site_code} - retrying with different approach")
            
            # Retry with a direct approach via dataretrieval functions
            try:
                # Try the site service again with a different method
                import requests
                url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={usgs_site_code}&siteOutput=expanded"
                
                logging.info(f"Fetching site info via direct request to {url}")
                response = requests.get(url)
                
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    
                    # Look for station name
                    for line in lines:
                        if usgs_site_code in line and not line.startswith('#'):
                            parts = line.split('\t')
                            if len(parts) > 2:
                                metadata['site_name'] = parts[2]
                                logging.info(f"Found site name from direct request: {metadata['site_name']}")
                                break
                                
                    # Look for datum information
                    for line in lines:
                        if 'NAVD88' in line:
                            metadata['datum'] = 'NAVD88'
                            logging.info(f"Found datum from direct request: NAVD88")
                            break
                        elif 'NGVD29' in line:
                            metadata['datum'] = 'NGVD29'
                            logging.info(f"Found datum from direct request: NGVD29")
                            break
                else:
                    logging.warning(f"Direct request failed with status code {response.status_code}")
            except Exception as e:
                logging.warning(f"Error in direct request approach: {e}")
    except Exception as e:
        logging.warning(f"Error retrieving site metadata: {e}")
    
    # Extract units and parameter description from data
    if not site_data.empty:
        # Get additional parameter metadata
        try:
            logging.info(f"Retrieving parameter info for code {parameter_code}")
            
            # Try getting data from a different endpoint
            try:
                param_info = nwis.get_record(
                    parameterCd=[parameter_code],
                    service='parameter'
                )
                
                if param_info is not None and not param_info.empty:
                    logging.info(f"Successfully retrieved parameter info with 'parameter' service")
                    # Process parameter info
                else:
                    logging.warning(f"No parameter info returned with 'parameter' service")
            except Exception as e:
                logging.warning(f"Error retrieving parameter info with 'parameter' service: {e}")
            
            # Apply parameter-specific defaults based on known codes
            param_descriptions = {
                '00065': 'Gage height',
                '62610': 'Water level, elevation, NAVD88',
                '62611': 'Water level, elevation, NGVD29',
                '62620': 'Water level, elevation, MSL'
            }
            
            if parameter_code in param_descriptions:
                metadata['parameter_desc'] = param_descriptions[parameter_code]
                logging.info(f"Using default parameter description: {metadata['parameter_desc']}")
                
                # Set units based on parameter code
                metadata['units'] = 'ft'  # Default units for these water level parameters
                logging.info(f"Using default units for {parameter_code}: feet")
                
                # Set datum based on parameter code
                if parameter_code == '62610':
                    metadata['datum'] = 'NAVD88'
                elif parameter_code == '62611':
                    metadata['datum'] = 'NGVD29'
                elif parameter_code == '62620':
                    metadata['datum'] = 'MSL'
                    
                if parameter_code != '00065':  # Only log datum for elevation parameters
                    logging.info(f"Setting datum to {metadata['datum']} based on parameter code {parameter_code}")
            
        except Exception as e:
            logging.warning(f"Error retrieving parameter metadata: {e}")
        
        # Apply parameter-specific defaults if needed
        if 'parameter_desc' not in metadata or not metadata['parameter_desc'] or metadata['parameter_desc'] == 'Unknown':
            # Set defaults based on parameter code
            param_descriptions = {
                '00065': 'Gage height',
                '62610': 'Water level, elevation, NAVD88',
                '62611': 'Water level, elevation, NGVD29',
                '62620': 'Water level, elevation, MSL'
            }
            if parameter_code in param_descriptions:
                metadata['parameter_desc'] = param_descriptions[parameter_code]
                logging.info(f"Using default parameter description: {metadata['parameter_desc']}")
        
        if 'units' not in metadata or not metadata['units'] or metadata['units'] == 'Unknown':
            # Set defaults based on parameter code
            if parameter_code == '00065':
                metadata['units'] = 'ft'
                logging.info("Using default units for 00065: feet")
            elif parameter_code in ['62610', '62611', '62620']:
                metadata['units'] = 'ft'
                logging.info(f"Using default units for {parameter_code}: feet")
        
        # Set datum based on parameter code if not already set
        if 'datum' not in metadata or not metadata['datum'] or metadata['datum'] == 'Unknown':
            if parameter_code == '62610':
                metadata['datum'] = 'NAVD88'
                logging.info("Setting datum to NAVD88 based on parameter code 62610")
            elif parameter_code == '62611':
                metadata['datum'] = 'NGVD29'
                logging.info("Setting datum to NGVD29 based on parameter code 62611")
            elif parameter_code == '62620':
                metadata['datum'] = 'MSL'
                logging.info("Setting datum to MSL based on parameter code 62620")
    
    # Ensure reasonable default values
    if 'site_name' not in metadata or not metadata['site_name']:
        metadata['site_name'] = f"USGS Site {usgs_site_code}"
        
    if 'parameter_desc' not in metadata or not metadata['parameter_desc']:
        metadata['parameter_desc'] = f"Parameter {parameter_code}"
        
    if 'datum' not in metadata or not metadata['datum']:
        metadata['datum'] = "Unknown datum"
    
    # Log final results
    logging.info(f"Final metadata: Site: {metadata['site_name']}, Param: {metadata['parameter_desc']}, Datum: {metadata['datum']}, Units: {metadata.get('units', 'Unknown')}")
    
    return metadata

def process_usgs_data(usgs_df, usgs_metadata, convert_to_meters=True):
    """
    Process USGS gauge data for comparison with GNSS-IR data.
    
    Args:
        usgs_df (pd.DataFrame): DataFrame containing raw USGS data
        usgs_metadata (dict): Metadata about the USGS gauge
        convert_to_meters (bool, optional): Whether to convert feet to meters if needed.
                                          Defaults to True.
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for comparison
    """
    if usgs_df.empty:
        logging.warning("No USGS data to process")
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = usgs_df.copy()
    
    # First, let's check if the DataFrame has the parameter code as a column
    param_code = usgs_metadata.get('parameter_code', None)
    if param_code and param_code in df.columns:
        # If parameter code exists as a column, use it directly
        df['usgs_value'] = df[param_code].astype(float)
        logging.info(f"Using column '{param_code}' directly from DataFrame")
    elif 'usgs_value' not in df.columns:
        # Try to find any value column that might contain the data
        value_cols = [col for col in df.columns if 'value' in str(col).lower() and not col.endswith('_cd')]
        if value_cols:
            df['usgs_value'] = df[value_cols[0]].astype(float)
            logging.info(f"Using column '{value_cols[0]}' as usgs_value")
        else:
            # Last resort: try any numeric column that isn't an ID or flag
            # Filter out non-numeric columns and those that are likely IDs or flags
            exclude_patterns = ['site', 'code', 'cd', 'flag', 'id', 'time', 'date', 'name']
            potential_cols = []
            
            for col in df.columns:
                # Skip columns that match exclude patterns
                if any(pattern in str(col).lower() for pattern in exclude_patterns):
                    continue
                    
                # Try to convert to numeric and check
                try:
                    if pd.to_numeric(df[col], errors='coerce').notna().any():
                        potential_cols.append(col)
                except (ValueError, TypeError):
                    pass
            
            if potential_cols:
                df['usgs_value'] = df[potential_cols[0]].astype(float)
                logging.info(f"Using column '{potential_cols[0]}' as usgs_value")
            else:
                logging.warning("Could not identify any suitable data column in USGS data")
                return pd.DataFrame()
    
    # Check if we need to convert units
    if convert_to_meters:
        # For parameter code 00065 (Gage height), always convert from feet to meters
        if param_code == '00065' or ('units' in usgs_metadata and usgs_metadata['units'].lower() == 'ft'):
            # Convert feet to meters
            if 'usgs_value' in df.columns:
                df['usgs_value_m'] = df['usgs_value'] * 0.3048
                logging.info("Converted water levels from feet to meters (00065 is always in feet)")
            else:
                logging.warning("Could not find 'usgs_value' column to convert from feet to meters")
        elif 'units' in usgs_metadata and usgs_metadata['units'].lower() == 'm':
            # Already in meters, just copy the column
            if 'usgs_value' in df.columns:
                df['usgs_value_m'] = df['usgs_value']
                logging.info("Water levels already in meters, no conversion needed")
            else:
                logging.warning("Could not find 'usgs_value' column")
        else:
            # Unknown units - default to assuming feet for water level parameters
            if 'usgs_value' in df.columns:
                df['usgs_value_m'] = df['usgs_value'] * 0.3048
                logging.info(f"Assuming feet for water level data and converting to meters (param code: {param_code})")
            else:
                logging.warning("Could not find 'usgs_value' column")
    
    # Create daily aggregated statistics
    if not df.empty and 'usgs_value_m' in df.columns:
        try:
            # Create a date column (just the date part of the datetime index)
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df['date'] = df.index.date
            elif 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime']).dt.date
            else:
                # Try to convert the first column that looks like a date
                date_cols = [col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()]
                if date_cols:
                    df['date'] = pd.to_datetime(df[date_cols[0]]).dt.date
                else:
                    logging.error("Could not find date column in USGS data")
                    return df
            
            # Group by date and calculate statistics
            daily_stats = df.groupby('date').agg({
                'usgs_value_m': ['count', 'mean', 'median', 'std', 'min', 'max']
            })
            
            # Flatten multi-level columns
            daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
            
            # Reset index to make date a column
            daily_stats = daily_stats.reset_index()
            
            # Convert date back to datetime for easier plotting
            daily_stats['datetime'] = pd.to_datetime(daily_stats['date'])
            
            logging.info(f"Created daily aggregated statistics for USGS data ({len(daily_stats)} days)")
            return daily_stats
        
        except Exception as e:
            logging.error(f"Error creating daily statistics: {e}")
            return df
    
    return df
