"""
Results Handler module for GNSS-IR processing.
This module combines daily reflector height files into annual datasets.
"""

import os
import glob
import logging
import pandas as pd
from pathlib import Path

def combine_daily_rh_results(station_id_4char_lower, year, daily_rh_base_dir, annual_results_dir):
    """
    Combine daily reflector height results into a single annual CSV file.
    
    Args:
        station_id_4char_lower (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        daily_rh_base_dir (str or Path): Base directory containing daily RH text files
        annual_results_dir (str or Path): Output directory for combined CSV
    
    Returns:
        Path or None: Path to the combined CSV file on success, None on failure
    """
    daily_rh_base_dir = Path(daily_rh_base_dir)
    annual_results_dir = Path(annual_results_dir)
    
    # Ensure the output directory exists
    annual_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output CSV path
    output_csv_path = annual_results_dir / f"{station_id_4char_lower.upper()}_{year}_combined_rh.csv"
    
    try:
        # Find all daily RH files - they should be named station_year_doy.txt
        file_pattern = f"{station_id_4char_lower}_{year}_*.txt"
        rh_files = sorted(daily_rh_base_dir.glob(file_pattern))
        
        if not rh_files:
            logging.warning(f"No RH files found matching pattern {file_pattern} in {daily_rh_base_dir}")
            return None
        
        logging.info(f"Found {len(rh_files)} RH files to combine")
        logging.info(f"Files: {[f.name for f in rh_files]}")
        
        # Initialize a list to store DataFrames
        all_data = []
        processing_errors = []
        
        # Process each RH file
        for rh_file in rh_files:
            try:
                # First check if the file format has comment lines at the top (indicated by % symbol)
                with open(rh_file, 'r') as f:
                    lines = f.readlines()
                
                # Skip empty files
                if not lines:
                    logging.warning(f"Empty file: {rh_file.name}, skipping")
                    processing_errors.append(f"{rh_file.name}: Empty file")
                    continue
                
                # Count the number of header lines (starting with %)
                header_lines = 0
                for line in lines:
                    if line.strip().startswith('%'):
                        header_lines += 1
                    else:
                        break
                
                logging.debug(f"Found {header_lines} header lines in {rh_file.name}")
                
                # Check if there's content after the header
                if header_lines >= len(lines):
                    logging.warning(f"No data after header in {rh_file.name}, skipping")
                    processing_errors.append(f"{rh_file.name}: No data after header")
                    continue
                
                # Try to extract column information from the header
                column_names = None
                column_descriptions = []
                
                # Look for the line with column descriptions (typically line 3)
                if header_lines >= 3:
                    # Line 3 (index 2) typically has descriptions like "% year, doy, RH, sat,UTCtime, Azim, Amp..."
                    desc_line = lines[2]
                    if desc_line.startswith('%'):
                        # Remove the % and split by commas
                        desc_parts = desc_line[1:].strip().split(',')
                        column_descriptions = [part.strip() for part in desc_parts]
                        logging.debug(f"Column descriptions: {column_descriptions}")
                
                # Now read the data, skipping the header lines
                try:
                    df = pd.read_csv(
                        rh_file, 
                        skiprows=header_lines,
                        delim_whitespace=True,
                        header=None
                    )
                except pd.errors.EmptyDataError:
                    logging.warning(f"No parseable data in {rh_file.name}, skipping")
                    processing_errors.append(f"{rh_file.name}: No parseable data")
                    continue
                
                # Check if we have any rows after parsing
                if df.empty:
                    logging.warning(f"Empty DataFrame after parsing {rh_file.name}, skipping")
                    processing_errors.append(f"{rh_file.name}: Empty DataFrame after parsing")
                    continue
                
                # Assign column names based on extracted descriptions if available
                if column_descriptions and len(column_descriptions) > 0:
                    # Assign names to columns based on the descriptions, up to the number of columns available
                    for i in range(min(len(column_descriptions), len(df.columns))):
                        df = df.rename(columns={i: column_descriptions[i]})
                    
                    logging.debug(f"Renamed columns based on descriptions: {df.columns.tolist()}")
                else:
                    # If no descriptions available, use generic column names
                    df = df.rename(columns=lambda x: f"Col{x+1}")
                    logging.debug(f"Using generic column names: {df.columns.tolist()}")
                
                # Extract the DOY from the filename
                filename_parts = rh_file.stem.split('_')
                if len(filename_parts) >= 3:
                    doy = filename_parts[-1]
                    try:
                        doy_int = int(doy)
                        # Add DOY as a column if it doesn't exist
                        if 'doy' not in df.columns:
                            df['doy'] = doy_int
                            logging.debug(f"Added DOY column with value {doy_int}")
                    except ValueError:
                        logging.warning(f"Could not extract valid DOY from filename: {rh_file.name}")
                else:
                    logging.warning(f"Filename format doesn't match expected pattern: {rh_file.name}")
                
                # Add the file to the list
                all_data.append(df)
                logging.info(f"Successfully processed {rh_file.name}, added {len(df)} rows")
                
            except Exception as e:
                logging.error(f"Error reading {rh_file}: {e}")
                processing_errors.append(f"{rh_file.name}: {str(e)}")
        
        if not all_data:
            logging.error("No valid data found in any RH files")
            if processing_errors:
                logging.error(f"Processing errors: {processing_errors}")
            return None
        
        # Concatenate all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Log the columns to help with debugging
        logging.info(f"Combined DataFrame columns: {combined_df.columns.tolist()}")
        logging.info(f"Combined DataFrame shape: {combined_df.shape}")
        
        # Sort by DOY if it exists
        if 'doy' in combined_df.columns:
            combined_df.sort_values('doy', inplace=True)
            logging.info(f"Sorted data by column: doy")
        
        # We'll handle date creation in the aggregation section below
        
        # Add date information (assuming year and doy columns exist)
        if 'year' in combined_df.columns and 'doy' in combined_df.columns:
            try:
                # Convert year and doy to datetime
                from datetime import datetime, timedelta
                combined_df['date'] = combined_df.apply(
                    lambda row: (datetime(int(row['year']), 1, 1) + timedelta(days=int(row['doy'])-1)).strftime('%Y-%m-%d'),
                    axis=1
                )
                logging.info("Added date column based on year and doy")
            except Exception as e:
                logging.warning(f"Could not add date column: {e}")

        # Perform daily aggregation
        try:
            # Ensure we have a date column for aggregation
            if 'date' not in combined_df.columns:
                logging.error("Cannot perform daily aggregation: date column is missing")
                # Save the raw combined data and return
                combined_df.to_csv(output_csv_path, index=False)
                logging.info(f"Raw combined data saved to {output_csv_path} with {len(combined_df)} total rows")
                return output_csv_path

            # Find the RH column
            rh_column = None
            for col in combined_df.columns:
                if col == 'RH' or col == 'rh':
                    rh_column = col
                    break
            
            if rh_column is None:
                logging.error("Cannot identify RH column for aggregation")
                # Save the raw combined data and return
                combined_df.to_csv(output_csv_path, index=False)
                logging.info(f"Raw combined data saved to {output_csv_path} with {len(combined_df)} total rows")
                return output_csv_path

            logging.info(f"Performing daily aggregation using '{rh_column}' column")

            # Group by date and calculate daily statistics
            daily_agg = combined_df.groupby('date').agg({
                rh_column: ['count', 'mean', 'median', 'std', 'min', 'max']
            })

            # Flatten the MultiIndex columns
            daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]

            # Rename columns to match expected format
            column_mapping = {
                f"{rh_column}_count": "rh_count",
                f"{rh_column}_mean": "rh_mean_m",
                f"{rh_column}_median": "rh_median_m",
                f"{rh_column}_std": "rh_std_m",
                f"{rh_column}_min": "rh_min_m",
                f"{rh_column}_max": "rh_max_m"
            }
            daily_agg.rename(columns=column_mapping, inplace=True)

            # Reset index to make date a regular column
            daily_agg.reset_index(inplace=True)

            # Add year and doy back from the date
            daily_agg['datetime'] = pd.to_datetime(daily_agg['date'])
            daily_agg['year'] = daily_agg['datetime'].dt.year
            daily_agg['doy'] = daily_agg['datetime'].dt.strftime('%j').astype(int)

            # Log the daily aggregation results
            daily_rows = len(daily_agg)
            original_rows = len(combined_df)
            logging.info(f"Daily aggregation complete: {original_rows} individual retrievals → {daily_rows} daily records")
            logging.info(f"Daily aggregated columns: {daily_agg.columns.tolist()}")

            # Save daily aggregated data to CSV
            daily_agg.to_csv(output_csv_path, index=False)
            logging.info(f"Daily aggregated data saved to {output_csv_path} with {len(daily_agg)} total rows")

            # Also save the original combined data for reference if needed
            raw_csv_path = output_csv_path.parent / f"{station_id_4char_lower.upper()}_{year}_combined_raw.csv"
            combined_df.to_csv(raw_csv_path, index=False)
            logging.info(f"Original combined data saved to {raw_csv_path} with {len(combined_df)} total rows")

            # Log any processing errors
            if processing_errors:
                logging.warning(f"Completed with {len(processing_errors)} processing errors: {processing_errors}")

            return output_csv_path
        except Exception as e:
            logging.error(f"Error during daily aggregation: {e}")
            # Save the raw combined data as fallback
            combined_df.to_csv(output_csv_path, index=False)
            logging.info(f"Raw combined data saved to {output_csv_path} with {len(combined_df)} total rows")
            
            # Log any processing errors
            if processing_errors:
                logging.warning(f"Completed with {len(processing_errors)} processing errors: {processing_errors}")
            
            return output_csv_path
    
    except Exception as e:
        logging.error(f"Error combining RH results: {e}")
        return None
