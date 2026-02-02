# ABOUTME: Data manager for RINEX file acquisition from AWS S3 and HTTP sources
# ABOUTME: Handles downloads, file validation, and local storage management

import os
import logging
import boto3
import botocore
import requests
from pathlib import Path
from urllib.parse import urlparse

def download_s3_file(s3_bucket, s3_key, local_target_path):
    """
    Download a file from AWS S3 to a local path.
    Supports public buckets without requiring AWS credentials.
    Falls back to HTTP download for public buckets.
    
    Args:
        s3_bucket (str): S3 bucket name
        s3_key (str): S3 key path
        local_target_path (str or Path): Local path to save the file
    
    Returns:
        bool: True if the download was successful, False otherwise
    """
    # Convert local_target_path to Path object if it's a string
    if isinstance(local_target_path, str):
        local_target_path = Path(local_target_path)
    
    # Ensure the target directory exists
    local_target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log the download attempt
    logging.info(f"Downloading S3 file: s3://{s3_bucket}/{s3_key} to {local_target_path}")
    
    # For public buckets like 'doi-gnss', let's prefer the direct HTTP method
    # to avoid credential-related issues
    if s3_bucket.lower() == 'doi-gnss' or s3_bucket.lower().startswith('public-'):
        logging.info(f"Using direct HTTP download for public bucket: {s3_bucket}")
        return download_s3_file_via_http(s3_bucket, s3_key, local_target_path)
    
    # Otherwise, try boto3 first
    try:
        # First try the boto3 client with anonymous credentials
        config = botocore.config.Config(signature_version=botocore.UNSIGNED)
        s3_client = boto3.client('s3', config=config)
        
        # Use get_object and stream the response for better anonymous access compatibility
        logging.info(f"Attempting S3 get_object and stream for: s3://{s3_bucket}/{s3_key}")
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        
        # Stream the content to the file
        with open(local_target_path, 'wb') as f:
            for chunk in response['Body'].iter_chunks(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        
        # Verify the file exists and has content
        if local_target_path.exists() and local_target_path.stat().st_size > 0:
            logging.info(f"Successfully downloaded file via get_object ({local_target_path.stat().st_size} bytes)")
            return True
        else:
            logging.error(f"File downloaded but appears to be empty or missing: {local_target_path}")
            return False
    
    except botocore.exceptions.NoCredentialsError:
        logging.warning(f"NoCredentialsError with boto3.get_object. Falling back to HTTP.")
        return download_s3_file_via_http(s3_bucket, s3_key, local_target_path)
    
    except botocore.exceptions.ClientError as e:
        # Handle specific S3 errors
        if e.response['Error']['Code'] == '404':
            logging.error(f"File not found in S3: s3://{s3_bucket}/{s3_key}")
            return False
        else:
            logging.warning(f"S3 ClientError with get_object: {e}. Falling back to HTTP.")
            return download_s3_file_via_http(s3_bucket, s3_key, local_target_path)
    
    except Exception as e:
        logging.warning(f"Unexpected error downloading S3 file via boto3.get_object: {e}. Trying HTTP download...")
        # Try downloading via direct HTTP request as a fallback
        return download_s3_file_via_http(s3_bucket, s3_key, local_target_path)

def download_s3_file_via_http(s3_bucket, s3_key, local_target_path):
    """
    Download a file from a public S3 bucket using HTTP requests.
    This is a fallback when boto3 fails due to credential issues.
    
    Args:
        s3_bucket (str): S3 bucket name
        s3_key (str): S3 key path
        local_target_path (str or Path): Local path to save the file
    
    Returns:
        bool: True if the download was successful, False otherwise
    """
    try:
        # Convert local_target_path to Path object if it's a string
        if isinstance(local_target_path, str):
            local_target_path = Path(local_target_path)
            
        # Ensure the target directory exists
        local_target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format the S3 HTTP URL
        url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        logging.info(f"Attempting HTTP download from: {url}")
        
        # Make the HTTP request with stream=True to handle large files
        response = requests.get(url, stream=True, timeout=60)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to the file
            with open(local_target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            
            # Verify the file exists and has content
            if local_target_path.exists() and local_target_path.stat().st_size > 0:
                logging.info(f"Successfully downloaded file via HTTP ({local_target_path.stat().st_size} bytes)")
                return True
            else:
                logging.error(f"File downloaded via HTTP but appears to be empty: {local_target_path}")
                return False
        else:
            # Try alternate URL format as fallback
            alternate_url = f"https://s3.amazonaws.com/{s3_bucket}/{s3_key}"
            if url != alternate_url:  # Only try if different
                logging.warning(f"HTTP download failed with status code {response.status_code}. Trying alternate URL format.")
                alt_response = requests.get(alternate_url, stream=True, timeout=60)
                
                if alt_response.status_code == 200:
                    # Write the content to the file
                    with open(local_target_path, 'wb') as f:
                        for chunk in alt_response.iter_content(chunk_size=8192):
                            if chunk: 
                                f.write(chunk)
                    
                    # Verify the file exists and has content
                    if local_target_path.exists() and local_target_path.stat().st_size > 0:
                        logging.info(f"Successfully downloaded file via alternate HTTP URL ({local_target_path.stat().st_size} bytes)")
                        return True
                    else:
                        logging.error(f"File downloaded via alternate HTTP URL but appears to be empty: {local_target_path}")
                        return False
                else:
                    logging.error(f"Both HTTP downloads failed. Original: {response.status_code}, Alternate: {alt_response.status_code}")
                    return False
            else:
                logging.error(f"HTTP download failed with status code {response.status_code}: {url}")
                return False
    
    except Exception as e:
        logging.error(f"Error during HTTP download: {e}")
        return False

def download_from_url(url, local_target_path):
    """
    Download a file from a URL to a local path.

    Args:
        url (str): Full URL to download from
        local_target_path (str or Path): Local path to save the file

    Returns:
        bool: True if the download was successful, False otherwise
    """
    try:
        if isinstance(local_target_path, str):
            local_target_path = Path(local_target_path)

        local_target_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading from URL: {url}")

        response = requests.get(url, stream=True, timeout=120)

        if response.status_code == 200:
            with open(local_target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if local_target_path.exists() and local_target_path.stat().st_size > 0:
                logging.info(f"Successfully downloaded file ({local_target_path.stat().st_size} bytes)")
                return True
            else:
                logging.error(f"File downloaded but appears to be empty: {local_target_path}")
                return False
        else:
            logging.error(f"HTTP download failed with status code {response.status_code}: {url}")
            return False

    except Exception as e:
        logging.error(f"Error during URL download: {e}")
        return False


def check_file_exists(file_path, min_size_bytes=0):
    """
    Check if a file exists and optionally check its size.
    
    Args:
        file_path (str or Path): Path to the file to check
        min_size_bytes (int, optional): Minimum size in bytes the file should have. Defaults to 0.
    
    Returns:
        bool: True if the file exists and meets size criteria, False otherwise
    """
    # Convert file_path to Path object if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if not file_path.exists():
        logging.debug(f"File does not exist: {file_path}")
        return False
    
    if not file_path.is_file():
        logging.debug(f"Path exists but is not a file: {file_path}")
        return False
    
    file_size = file_path.stat().st_size
    if file_size < min_size_bytes:
        logging.debug(f"File is too small: {file_path} ({file_size} bytes < {min_size_bytes} bytes required)")
        return False
    
    return True

def clean_temp_files(directory_path, pattern="*", recursive=False, older_than_days=None):
    """
    Clean temporary files matching a pattern from a directory.
    
    Args:
        directory_path (str or Path): Directory to clean
        pattern (str, optional): Glob pattern for files to delete. Defaults to "*".
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        older_than_days (int, optional): Only delete files older than this many days. Defaults to None.
    
    Returns:
        tuple: (int, list) - Number of files deleted and list of errors encountered
    """
    import time
    from datetime import datetime, timedelta
    
    # Convert directory_path to Path object if it's a string
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)
    
    if not directory_path.exists() or not directory_path.is_dir():
        logging.warning(f"Directory does not exist or is not a directory: {directory_path}")
        return 0, [f"Directory does not exist or is not a directory: {directory_path}"]
    
    # Calculate cutoff time if older_than_days is specified
    cutoff_time = None
    if older_than_days is not None:
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
    
    # Find files to delete
    glob_pattern = "**/" + pattern if recursive else pattern
    files_to_delete = list(directory_path.glob(glob_pattern))
    
    # Filter by age if needed
    if cutoff_time is not None:
        files_to_delete = [f for f in files_to_delete if f.is_file() and 
                           datetime.fromtimestamp(f.stat().st_mtime) < cutoff_time]
    else:
        files_to_delete = [f for f in files_to_delete if f.is_file()]
    
    # Delete files
    deleted_count = 0
    errors = []
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            deleted_count += 1
            logging.debug(f"Deleted file: {file_path}")
        except Exception as e:
            error_msg = f"Error deleting {file_path}: {e}"
            errors.append(error_msg)
            logging.error(error_msg)
    
    logging.info(f"Cleaned {deleted_count} files from {directory_path}")
    if errors:
        logging.warning(f"Encountered {len(errors)} errors while cleaning files")
    
    return deleted_count, errors
