# ABOUTME: Time lag correlation analysis for GNSS-IR vs reference data
# ABOUTME: Computes optimal lag using cross-correlation for tidal propagation delays

import logging
import numpy as np
import pandas as pd

def calculate_time_lag_correlation(series1, series2, max_lag_days=10):
    """
    Calculate the time lag with maximum correlation between two time series.
    
    Args:
        series1 (pd.Series): First time series
        series2 (pd.Series): Second time series
        max_lag_days (int): Maximum lag to consider in days (both directions)
    
    Returns:
        tuple: (optimal_lag_days, max_correlation, confidence_level, all_correlations)
    """
    try:
        # For shorter time series, limit the maximum lag to a reasonable value
        series_length = min(len(series1), len(series2))
        
        # If we have a very short time series, limit the maximum lag
        # to ensure we have at least 3 overlapping points
        effective_max_lag = min(max_lag_days, series_length - 3)
        if effective_max_lag < max_lag_days:
            logging.info(f"Limited maximum lag to {effective_max_lag} days to ensure at least 3 overlapping points")
        
        if series_length < 4:  # Need at least 4 points to have 2+ points overlap with a lag
            logging.warning("Time series too short for meaningful lag analysis (less than 4 points)")
            return None, None, "Low", {}
        
        # First, ensure both series are demeaned (zero mean) and normalized
        series1_mean = np.mean(series1)
        series1_std = np.std(series1) if np.std(series1) != 0 else 1
        series1_norm = (series1 - series1_mean) / series1_std
        
        series2_mean = np.mean(series2)
        series2_std = np.std(series2) if np.std(series2) != 0 else 1
        series2_norm = (series2 - series2_mean) / series2_std
        
        # Convert to numpy arrays
        s1 = np.array(series1_norm)
        s2 = np.array(series2_norm)
        
        # Initialize variables to track best lag
        best_lag = 0
        best_corr = -np.inf
        all_correlations = {}
        num_overlap_points = {}
        
        # Calculate correlation for different lags up to effective_max_lag in both directions
        for lag in range(-effective_max_lag, effective_max_lag + 1):
            # Calculate the overlap size at this lag
            if lag < 0:
                overlap_size = min(len(s1) + lag, len(s2))
            elif lag > 0:
                overlap_size = min(len(s1) - lag, len(s2))
            else:  # lag == 0
                overlap_size = min(len(s1), len(s2))
            
            # Skip if not enough overlap
            if overlap_size < 3:  # Require at least 3 overlapping points for correlation
                logging.debug(f"Skipping lag {lag} - only {overlap_size} overlapping points")
                all_correlations[lag] = None
                num_overlap_points[lag] = overlap_size
                continue
            
            # Shift the arrays and calculate correlation
            if lag < 0:
                # s2 is ahead of s1 by |lag| steps
                x1 = s1[:lag]
                x2 = s2[-lag:]
            elif lag > 0:
                # s1 is ahead of s2 by lag steps
                x1 = s1[lag:]
                x2 = s2[:-lag]
            else:
                # No lag
                x1 = s1
                x2 = s2
            
            # Calculate correlation
            try:
                corr = np.corrcoef(x1, x2)[0, 1]
                all_correlations[lag] = corr
                num_overlap_points[lag] = overlap_size
                
                # Update best lag if correlation is higher
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag
                    
                logging.debug(f"Lag {lag}: corr = {corr:.4f}, overlap = {overlap_size} points")
            except Exception as e:
                logging.error(f"Error calculating correlation for lag {lag}: {e}")
                all_correlations[lag] = None
                num_overlap_points[lag] = overlap_size
        
        # Determine confidence level based on number of overlapping points
        # and correlation strength
        overlap_at_best_lag = num_overlap_points.get(best_lag, 0)
        
        if overlap_at_best_lag >= 7 and abs(best_corr) > 0.8:
            confidence = "High"
        elif overlap_at_best_lag >= 5 and abs(best_corr) > 0.7:
            confidence = "Medium"
        elif overlap_at_best_lag >= 3 and abs(best_corr) > 0.5:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        logging.info(f"Optimal time lag: {best_lag} days with correlation {best_corr:.4f} " +
                   f"(confidence: {confidence}, {overlap_at_best_lag} overlapping points)")
        
        return best_lag, best_corr, confidence, all_correlations
    
    except Exception as e:
        logging.error(f"Error calculating time lag correlation: {e}")
        return None, None, "Error", {}

def create_lag_adjusted_series(series1, series2, lag_days):
    """
    Create lag-adjusted versions of two time series.
    
    Args:
        series1 (pd.Series): First time series
        series2 (pd.Series): Second time series
        lag_days (int): Lag in days (positive means series1 leads series2)
    
    Returns:
        tuple: (adjusted_series1, adjusted_series2)
    """
    try:
        if lag_days == 0:
            return series1, series2
        
        # Convert to numpy arrays
        s1 = np.array(series1)
        s2 = np.array(series2)
        
        # Adjust for lag
        if lag_days > 0:
            # series1 leads series2 by lag_days
            s1_adj = s1[lag_days:]
            s2_adj = s2[:-lag_days]
        else:
            # series2 leads series1 by abs(lag_days)
            lag_days = abs(lag_days)
            s1_adj = s1[:-lag_days]
            s2_adj = s2[lag_days:]
        
        # Convert back to series
        s1_adj_series = pd.Series(s1_adj)
        s2_adj_series = pd.Series(s2_adj)
        
        logging.info(f"Created lag-adjusted series with lag {lag_days} days")
        return s1_adj_series, s2_adj_series
    
    except Exception as e:
        logging.error(f"Error creating lag-adjusted series: {e}")
        return series1, series2
