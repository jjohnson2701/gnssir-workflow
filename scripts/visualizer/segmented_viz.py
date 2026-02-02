# ABOUTME: Segmented correlation visualization for monthly/seasonal analysis
# ABOUTME: Creates correlation heatmaps, comparison grids, and time series by segment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def plot_segment_correlations(segment_correlations: Dict[str, Optional[float]], 
                             title: str = "Segmented Correlation Analysis",
                             ylabel: str = "Correlation Coefficient", 
                             figsize: Tuple[int, int] = (10, 6),
                             color: str = 'steelblue',
                             highlight_threshold: Optional[float] = None,
                             highlight_color: str = 'indianred',
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a bar chart of segment correlations.
    
    Parameters:
    -----------
    segment_correlations : Dictionary with segment names as keys and correlation values as values
    title : Plot title
    ylabel : Y-axis label
    figsize : Figure size as (width, height) tuple
    color : Bar color
    highlight_threshold : Correlation absolute value threshold for highlighting (e.g., 0.7)
    highlight_color : Color for highlighted bars
    save_path : Optional path to save the figure
    
    Returns: Matplotlib Figure object
    """
    # Filter out None values
    filtered_correlations = {k: v for k, v in segment_correlations.items() if v is not None and not pd.isna(v)}
    
    if not filtered_correlations:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid correlation data available", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    segments = list(filtered_correlations.keys())
    correlations = list(filtered_correlations.values())
    
    # Create bar colors
    bar_colors = []
    for corr in correlations:
        if highlight_threshold is not None and abs(corr) >= highlight_threshold:
            bar_colors.append(highlight_color)
        else:
            bar_colors.append(color)
    
    # Create bar chart
    bars = ax.bar(segments, correlations, color=bar_colors)
    
    # Add horizontal line at r=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add correlation values above/below bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        position = height + 0.02 if height >= 0 else height - 0.08
        ax.text(bar.get_x() + bar.get_width()/2., position,
                f'{corr:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(min(correlations) - 0.1, -0.1), max(max(correlations) + 0.1, 1.0))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_segment_comparison_grid(segment_data: Dict[str, pd.DataFrame],
                               gnss_col: str = 'wse_ellips_m_demeaned',
                               usgs_col: str = 'usgs_value_m_median_demeaned',
                               segment_correlations: Optional[Dict[str, float]] = None,
                               max_cols: int = 3,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a grid of scatter plots for each segment's data.
    
    Parameters:
    -----------
    segment_data : Dictionary with segment names as keys and DataFrames as values
    gnss_col : Column name for GNSS-IR data
    usgs_col : Column name for USGS data
    segment_correlations : Dictionary with segment names as keys and correlation values
    max_cols : Maximum number of columns in the grid
    figsize : Figure size as (width, height) tuple
    save_path : Optional path to save the figure
    
    Returns: Matplotlib Figure object
    """
    # Filter out segments with no data
    filtered_segments = {k: v for k, v in segment_data.items() 
                        if v is not None and len(v) > 0 and gnss_col in v.columns and usgs_col in v.columns}
    
    n_segments = len(filtered_segments)
    
    if n_segments == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid segment data available", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Calculate grid dimensions
    n_cols = min(n_segments, max_cols)
    n_rows = (n_segments + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Ensure axes is a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Use colors based on correlation strength
    cmap = plt.cm.get_cmap('coolwarm')
    
    # Plot each segment
    for i, (segment_name, df) in enumerate(filtered_segments.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get correlation
        if segment_correlations is not None and segment_name in segment_correlations:
            corr = segment_correlations[segment_name]
        else:
            # Filter out NaN values before correlation
            valid_mask = df[gnss_col].notna() & df[usgs_col].notna()
            if valid_mask.sum() > 1:
                corr = df.loc[valid_mask, gnss_col].corr(df.loc[valid_mask, usgs_col])
            else:
                corr = np.nan
        
        # Color based on correlation strength
        point_color = 'steelblue'
        line_color = 'red'
        
        if corr is not None and not pd.isna(corr):
            # Use coolwarm colormap: blue for negative, red for positive
            point_color = cmap((corr + 1) / 2)  # Scale from [-1, 1] to [0, 1]
        
        # Scatter plot
        ax.scatter(df[gnss_col], df[usgs_col], alpha=0.7, color=point_color)
        
        # Add best fit line
        if len(df) > 1:
            x = df[gnss_col].values
            y = df[usgs_col].values
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            if len(x) > 1:
                poly = np.polyfit(x, y, 1)
                p = np.poly1d(poly)
                x_sorted = np.sort(x)
                ax.plot(x_sorted, p(x_sorted), '--', color=line_color, linewidth=1)
        
        # Add title with correlation
        corr_str = f"{corr:.2f}" if corr is not None and not pd.isna(corr) else "N/A"
        n_points = df[gnss_col].notna() & df[usgs_col].notna()
        n_points = n_points.sum()
        ax.set_title(f"{segment_name} (r = {corr_str}, n = {n_points})")
        
        # Add x and y labels for the first column and last row only
        if col == 0:
            ax.set_ylabel('USGS Water Level')
        if row == n_rows - 1:
            ax.set_xlabel('GNSS-IR WSE')
    
    # Hide unused subplots
    for i in range(n_segments, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_time_series_by_segment(df: pd.DataFrame,
                              gnss_col: str,
                              usgs_col: str,
                              segments_dict: Dict[str, Union[Tuple, List]],
                              segment_correlations: Dict[str, float],
                              station_name: str = '',
                              figsize: Tuple[int, int] = (12, 8),
                              demeaned: bool = True,
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot time series data colored by segment with correlation annotations.
    
    Parameters:
    -----------
    df : DataFrame containing both GNSS-IR and USGS data with date column
    gnss_col : Column name for GNSS-IR data
    usgs_col : Column name for USGS data
    segments_dict : Dictionary with segment names as keys and segment criteria as values
    segment_correlations : Dictionary with segment names as keys and correlation values
    station_name : Station name for plot title
    figsize : Figure size as (width, height) tuple
    demeaned : Whether the data is demeaned (affects plot title)
    save_path : Optional path to save the figure
    
    Returns: Matplotlib Figure object
    """
    # Ensure df has datetime index for filtering
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        if 'date' in df_copy.columns:
            df_copy.index = pd.to_datetime(df_copy['date'])
        elif 'datetime' in df_copy.columns:
            df_copy.index = pd.to_datetime(df_copy['datetime'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot full time series in light gray for context
    ax1.plot(df_copy.index, df_copy[gnss_col], color='lightgray', linewidth=1, label='Full Period')
    ax2.plot(df_copy.index, df_copy[usgs_col], color='lightgray', linewidth=1, label='Full Period')
    
    # Define colors for segments
    n_segments = len(segments_dict)
    colors = plt.cm.tab10(np.linspace(0, 1, n_segments))
    
    # Plot each segment with different color
    for i, (segment_name, date_criteria) in enumerate(segments_dict.items()):
        # Filter data for this segment
        segment_df = df_copy.copy()
        segment_df = segment_df.loc[df_copy.index.isin(filter_by_segment(df_copy, date_criteria).index)]
        
        if len(segment_df) == 0:
            continue
        
        # Get correlation value
        corr = segment_correlations.get(segment_name)
        corr_str = f"{corr:.2f}" if corr is not None and not pd.isna(corr) else "N/A"
        
        # Plot segment data
        color = colors[i]
        ax1.plot(segment_df.index, segment_df[gnss_col], 'o-', color=color, 
                 label=f"{segment_name} (r = {corr_str})")
        ax2.plot(segment_df.index, segment_df[usgs_col], 'o-', color=color)
    
    # Format axes
    ax1.set_title(f"{station_name} GNSS-IR vs USGS Segmented Analysis" + 
                 (" (Demeaned)" if demeaned else ""))
    ax1.set_ylabel('GNSS-IR WSE (m)')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('USGS Water Level (m)')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_heatmap_correlation_matrix(segment_correlations: Dict[str, float],
                                  title: str = "Correlation Heatmap",
                                  figsize: Tuple[int, int] = (8, 6),
                                  cmap: str = 'coolwarm',
                                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a heatmap of segment correlations.
    
    Parameters:
    -----------
    segment_correlations : Dictionary with segment names as keys and correlation values
    title : Plot title
    figsize : Figure size as (width, height) tuple
    cmap : Colormap for heatmap
    save_path : Optional path to save the figure
    
    Returns: Matplotlib Figure object
    """
    # Filter out None and NaN values
    filtered_correlations = {k: v for k, v in segment_correlations.items() 
                           if v is not None and not pd.isna(v)}
    
    if not filtered_correlations:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid correlation data available", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Convert dictionary to Series
    corr_series = pd.Series(filtered_correlations)
    
    # Reshape into a DataFrame for heatmap
    # Use 1x5 grid if 5 or fewer segments, otherwise reshape to a more square grid
    n_segments = len(filtered_correlations)
    
    if n_segments <= 5:
        corr_df = pd.DataFrame(corr_series.values.reshape(1, -1),
                              index=['Correlation'],
                              columns=corr_series.index)
    else:
        # Calculate a reasonable grid shape
        n_cols = int(np.ceil(np.sqrt(n_segments)))
        n_rows = int(np.ceil(n_segments / n_cols))
        
        # Pad with NaNs to fill the grid
        padded_values = np.pad(corr_series.values, 
                             (0, n_rows * n_cols - n_segments),
                             'constant', 
                             constant_values=np.nan)
        
        # Reshape and create DataFrame
        corr_df = pd.DataFrame(padded_values.reshape(n_rows, n_cols),
                             index=[f'Row {i+1}' for i in range(n_rows)],
                             columns=[corr_series.index[i] if i < len(corr_series) else '' 
                                     for i in range(n_cols)])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr_df, annot=True, cmap=cmap, vmin=-1, vmax=1, 
               linewidths=0.5, ax=ax, fmt='.2f', center=0)
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig