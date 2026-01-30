"""
Visualizer module for GNSS-IR processing.
Provides functions for generating plots and visualizations.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def plot_annual_rh_timeseries(combined_rh_csv_path, station_name, year, annual_results_dir):
    """
    Generate an annual reflector height time series plot.
    
    Args:
        combined_rh_csv_path (str or Path): Path to the combined reflector heights CSV file
        station_name (str): Station name (e.g., "FORA")
        year (int or str): Year for the plot
        annual_results_dir (str or Path): Directory for the annual results
        
    Returns:
        Path: Path to the generated plot file, or None if the plot generation failed
    """
    # Convert paths to Path objects
    combined_rh_csv_path = Path(combined_rh_csv_path)
    annual_results_dir = Path(annual_results_dir)
    
    # Ensure annual_results_dir exists
    annual_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output plot path
    plot_path = annual_results_dir / f"{station_name}_{year}_annual_waterlevel.png"
    
    try:
        # Load the combined RH data
        df = pd.read_csv(combined_rh_csv_path)
        
        # Check if required columns exist
        required_columns = ['date', 'rh_median_m']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns in combined RH CSV: {missing_columns}")
            return None
        
        # Convert date string to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot reflector height median with error bars if available
        if 'rh_median_m' in df.columns and 'rh_std_m' in df.columns:
            plt.errorbar(df['date'], df['rh_median_m'], yerr=df['rh_std_m'], 
                       fmt='o-', color='blue', ecolor='lightblue', 
                       elinewidth=1, capsize=2, label='RH median with STD')
        else:
            plt.plot(df['date'], df['rh_median_m'], 'o-', color='blue', label='RH median')
        
        # Add titles and labels
        plt.title(f"{station_name} Reflector Height for {year}")
        plt.xlabel("Date")
        plt.ylabel("Reflector Height (m)")
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Add annotation with data points count
        plt.annotate(f"Total days: {len(df)}", 
                   xy=(0.02, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add legend
        plt.legend()
        
        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        logging.info(f"Annual reflector height plot saved to {plot_path}")
        return plot_path
    
    except Exception as e:
        logging.error(f"Error generating annual reflector height plot: {e}")
        return None

def plot_comparison_timeseries(daily_gnssir_rh_df, daily_usgs_gauge_df, station_name, 
                             usgs_gauge_info, output_plot_path, 
                             gnssir_rh_col='rh_median_m', usgs_wl_col='usgs_value_m_median',
                             compare_demeaned=False):
    """
    Generate a comparison plot between GNSS-IR reflector heights and USGS gauge data.
    
    Args:
        daily_gnssir_rh_df (pd.DataFrame): DataFrame containing daily GNSS-IR reflector heights
        daily_usgs_gauge_df (pd.DataFrame): DataFrame containing daily USGS gauge data
        station_name (str): Station name (e.g., "FORA")
        usgs_gauge_info (dict): Dictionary containing USGS gauge metadata
        output_plot_path (str or Path): Path to save the output plot
        gnssir_rh_col (str, optional): Column name for GNSS-IR reflector height data.
                                     Defaults to 'rh_median_m'.
        usgs_wl_col (str, optional): Column name for USGS water level data.
                                   Defaults to 'usgs_value_m_median'.
        compare_demeaned (bool, optional): Whether to plot demeaned data. Defaults to False.
    
    Returns:
        Path: Path to the generated plot file, or None if the plot generation failed
    """
    # Convert output_plot_path to Path object
    output_plot_path = Path(output_plot_path)
    
    # Ensure parent directory exists
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract gauge info
        gauge_site_code = usgs_gauge_info.get('site_code', 'Unknown')
        gauge_site_name = usgs_gauge_info.get('site_name', 'Unknown')
        gauge_datum = usgs_gauge_info.get('datum', 'Unknown datum')
        
        # Convert date columns to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(daily_gnssir_rh_df['date']):
            daily_gnssir_rh_df['date'] = pd.to_datetime(daily_gnssir_rh_df['date'])
        
        if not pd.api.types.is_datetime64_any_dtype(daily_usgs_gauge_df['date']):
            daily_usgs_gauge_df['date'] = pd.to_datetime(daily_usgs_gauge_df['date'])
        
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Set plot title based on comparison type
        if compare_demeaned:
            plot_title = f"{station_name} GNSS-IR vs USGS Gauge {gauge_site_code} (Demeaned)"
        else:
            plot_title = f"{station_name} GNSS-IR vs USGS Gauge {gauge_site_code}"
        
        # Plot the data
        if compare_demeaned:
            # Demeaned plot (single Y-axis)
            # Check if rh_count column exists in the dataframe for color coding
            if 'rh_count' in daily_gnssir_rh_df.columns:
                # Normalize rh_count for colormap
                min_c = daily_gnssir_rh_df['rh_count'].min()
                max_c = daily_gnssir_rh_df['rh_count'].max()
                norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
                cmap = plt.cm.viridis  # You can use other colormaps: plasma, coolwarm, etc.
                
                # Create scatter plot with color based on rh_count
                scatter = ax1.scatter(
                    daily_gnssir_rh_df['date'],
                    daily_gnssir_rh_df[gnssir_rh_col],
                    c=daily_gnssir_rh_df['rh_count'],
                    cmap=cmap,
                    norm=norm,
                    s=50,  # marker size
                    alpha=0.7,
                    zorder=10,
                    label=f"{station_name} GNSS-IR (Color by Count)"
                )
                
                # Add a light connecting line if desired
                ax1.plot(daily_gnssir_rh_df['date'], daily_gnssir_rh_df[gnssir_rh_col], 
                       '-', color='blue', alpha=0.3, zorder=5)
                
                # Add a colorbar
                cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical')
                cbar.set_label('Daily RH Retrieval Count')
            else:
                # Fallback if rh_count is not available
                ax1.plot(daily_gnssir_rh_df['date'], daily_gnssir_rh_df[gnssir_rh_col], 
                       'o-', color='blue', label=f'GNSS-IR {gnssir_rh_col}')
            
            # Plot USGS data
            ax1.plot(daily_usgs_gauge_df['date'], daily_usgs_gauge_df[usgs_wl_col], 
                   's-', color='red', label=f'USGS {usgs_wl_col}')
            
            ax1.set_ylabel("Demeaned Value (m)")
        else:
            # Regular plot (dual Y-axes)
            # Plot GNSS-IR data on primary Y-axis
            color1 = 'blue'
            ax1.set_ylabel(f'GNSS-IR {gnssir_rh_col} (m)', color=color1)
            
            if 'rh_count' in daily_gnssir_rh_df.columns:
                # Normalize rh_count for colormap
                min_c = daily_gnssir_rh_df['rh_count'].min()
                max_c = daily_gnssir_rh_df['rh_count'].max()
                norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
                cmap = plt.cm.viridis  # You can use other colormaps: plasma, coolwarm, etc.
                
                # Create scatter plot with color based on rh_count
                scatter = ax1.scatter(
                    daily_gnssir_rh_df['date'],
                    daily_gnssir_rh_df[gnssir_rh_col],
                    c=daily_gnssir_rh_df['rh_count'],
                    cmap=cmap,
                    norm=norm,
                    s=50,  # marker size
                    alpha=0.7,
                    zorder=10,
                    label=f"{station_name} GNSS-IR (Color by Count)"
                )
                
                # Add a light connecting line if desired
                ax1.plot(daily_gnssir_rh_df['date'], daily_gnssir_rh_df[gnssir_rh_col], 
                       '-', color=color1, alpha=0.3, zorder=5)
                
                # Add a colorbar
                cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical')
                cbar.set_label('Daily RH Retrieval Count')
            else:
                # Fallback if rh_count is not available
                ax1.plot(daily_gnssir_rh_df['date'], daily_gnssir_rh_df[gnssir_rh_col], 
                       'o-', color=color1, label=f'GNSS-IR {gnssir_rh_col}')
                
            ax1.tick_params(axis='y', labelcolor=color1)
            
            # Create secondary Y-axis for USGS data
            ax2 = ax1.twinx()
            color2 = 'red'
            ax2.set_ylabel(f'USGS {usgs_wl_col} (m, {gauge_datum})', color=color2)
            ax2.plot(daily_usgs_gauge_df['date'], daily_usgs_gauge_df[usgs_wl_col], 
                    's-', color=color2, label=f'USGS {usgs_wl_col}')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Create a combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Calculate correlation if data overlaps and add to plot title
        merged_df = pd.merge(
            daily_gnssir_rh_df[['date', gnssir_rh_col]],
            daily_usgs_gauge_df[['date', usgs_wl_col]],
            on='date',
            how='inner'
        )
        
        if len(merged_df) >= 2:  # Need at least 2 points for correlation
            correlation = merged_df[gnssir_rh_col].corr(merged_df[usgs_wl_col])
            plot_title += f"\nCorrelation: {correlation:.4f} (n={len(merged_df)})"
        
        # Add time lag info if available
        if 'time_lag_days' in usgs_gauge_info:
            lag_days = usgs_gauge_info['time_lag_days']
            lag_corr = usgs_gauge_info.get('lag_correlation', 'N/A')
            lag_conf = usgs_gauge_info.get('lag_confidence', 'N/A')
            
            if isinstance(lag_corr, (int, float)) and not pd.isna(lag_corr):
                lag_corr_str = f"{lag_corr:.4f}"
            else:
                lag_corr_str = str(lag_corr)
                
            plot_title += f"\nTime Lag: {lag_days} days, Lagged Correlation: {lag_corr_str}, Confidence: {lag_conf}"
        
        # Complete plot formatting
        ax1.set_xlabel('Date')
        ax1.set_title(plot_title)
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Add metadata annotation
        metadata_text = f"USGS Site: {gauge_site_code}\n"
        metadata_text += f"USGS Site Name: {gauge_site_name}\n"
        metadata_text += f"USGS Datum: {gauge_datum}\n"
        metadata_text += f"Overlapping days: {len(merged_df)}"
        
        plt.annotate(metadata_text, 
                   xy=(0.02, 0.02), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   verticalalignment='bottom')
        
        # If this is a demeaned plot, add the legend (for dual Y-axis plots, legend is added earlier)
        if compare_demeaned:
            ax1.legend()
        
        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=100)
        plt.close()
        
        logging.info(f"Comparison plot saved to {output_plot_path}")
        return output_plot_path
    
    except Exception as e:
        logging.error(f"Error generating comparison plot: {e}")
        return None
