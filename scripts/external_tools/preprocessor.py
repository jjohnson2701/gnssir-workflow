"""
Preprocessor module for GNSS-IR processing.
Provides functions for preprocessing RINEX files.
"""

import os
import logging
import subprocess
from pathlib import Path

def convert_rinex3_to_rinex2(gfzrnx_exe_path, rinex3_file_path, rinex2_output_dir, 
                           station_4char_lower, year, doy):
    """
    Convert RINEX 3 file to RINEX 2.11 format using gfzrnx.
    
    Args:
        gfzrnx_exe_path (str): Path to gfzrnx executable
        rinex3_file_path (str or Path): Path to RINEX 3 file
        rinex2_output_dir (str or Path): Directory to output RINEX 2.11 file
        station_4char_lower (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        doy (int): Day of year
    
    Returns:
        Path: Path to the converted RINEX 2.11 file if successful, None otherwise
    """
    # Convert paths to Path objects
    rinex3_file_path = Path(rinex3_file_path)
    rinex2_output_dir = Path(rinex2_output_dir)
    
    # Ensure RINEX 3 file exists
    if not rinex3_file_path.exists():
        logging.error(f"RINEX 3 file not found: {rinex3_file_path}")
        return None
    
    # Ensure output directory exists
    rinex2_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format the doy with zero-padding
    doy_padded = f"{doy:03d}"
    
    # Determine the last two digits of the year for RINEX 2.11 filename
    yy = str(year)[-2:]
    
    # Define expected RINEX 2.11 filename and path
    rinex2_filename = f"{station_4char_lower}{doy_padded}0.{yy}o"
    rinex2_path = rinex2_output_dir / rinex2_filename
    
    # Log the conversion step
    logging.info(f"Converting RINEX 3 file {rinex3_file_path} to RINEX 2.11 format at {rinex2_path}")
    
    try:
        # Construct the gfzrnx command
        cmd = [
            gfzrnx_exe_path,
            "-finp", str(rinex3_file_path),
            "-fout", str(rinex2_path),
            "-vo", "2",
            "-f"
        ]
        
        # Calculate timeout based on file size (larger files need more time)
        file_size_mb = rinex3_file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:  # Files larger than 500MB
            timeout_seconds = min(1800, file_size_mb * 2)  # Max 30 minutes, 2 seconds per MB
        else:
            timeout_seconds = 300  # Default 5 minutes for smaller files
        
        logging.info(f"Using timeout of {timeout_seconds} seconds for {file_size_mb:.1f}MB file")
        
        # Run the command
        logging.debug(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False  # Don't raise an exception on non-zero return code
        )
        
        # Check if the process was successful and the output file exists
        if process.returncode == 0 and rinex2_path.exists():
            logging.info(f"Successfully converted RINEX 3 to RINEX 2.11: {rinex2_path}")
            return rinex2_path
        else:
            # Log error information
            logging.error(f"gfzrnx conversion failed with return code {process.returncode}")
            logging.error(f"stdout: {process.stdout}")
            logging.error(f"stderr: {process.stderr}")
            
            # Check if output file exists despite error
            if rinex2_path.exists():
                logging.warning(f"Output file exists despite error: {rinex2_path}")
                return rinex2_path
            
            return None
    
    except subprocess.TimeoutExpired:
        logging.error(f"gfzrnx conversion timed out after {timeout_seconds} seconds for file size {file_size_mb:.1f}MB")
        return None
    except Exception as e:
        logging.error(f"Exception during RINEX conversion: {e}")
        return None
