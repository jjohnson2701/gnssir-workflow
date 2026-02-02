# ABOUTME: Diagnostic comparison plots for GNSS-IR vs reference data analysis
# ABOUTME: Provides quality diagnostics, seasonal investigation, and outlier detection

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats

from .base import ensure_output_dir, PLOT_COLORS

def create_comparison_plot(df, station_name, year, output_path):
    """
    Create improved GNSS-IR vs USGS comparison plot with better styling
    
    Args:
        df: DataFrame with both GNSS-IR and USGS data
        station_name: Station name (e.g., "FORA")
        year: Year for the data
        output_path: Path to save the plot
    
    Returns:
        Tuple of (overall_correlation, demeaned_correlation)
    """
    # Define better color scheme
    colors = PLOT_COLORS
    
    # Create figure with higher DPI for better quality
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    fig.suptitle(f'{station_name} GNSS-IR vs USGS Water Level Comparison ({year})', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Prepare data
    df_clean = df.dropna(subset=['wse_ellips_m', 'usgs_value_m_median'])
    
    if len(df_clean) == 0:
        logging.error("No valid data for comparison")
        return None, None
    
    # Convert index to datetime if it's not already
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean.index = pd.to_datetime(df_clean.index)
    
    # Plot 1: Raw comparison with dual y-axis
    ax1_twin = ax1.twinx()
    
    # GNSS-IR data (left axis)
    line1 = ax1.plot(df_clean.index, df_clean['wse_ellips_m'], 
                     color=colors['gnss'], linewidth=2.5, 
                     marker='o', markersize=4, alpha=0.8,
                     label='GNSS-IR WSE')
    
    # USGS data (right axis) - improved styling
    line2 = ax1_twin.plot(df_clean.index, df_clean['usgs_value_m_median'], 
                          color=colors['usgs'], linewidth=2.5,
                          marker='s', markersize=3, alpha=0.8,
                          label='USGS Water Level')
    
    # Styling for top plot
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('GNSS-IR WSE (m)', color=colors['gnss'], fontweight='bold')
    ax1_twin.set_ylabel('USGS Water Level (m)', color=colors['usgs'], fontweight='bold')
    
    ax1.tick_params(axis='y', labelcolor=colors['gnss'])
    ax1_twin.tick_params(axis='y', labelcolor=colors['usgs'])
    
    # Add grid
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    
    # Format dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
    
    # Add correlation text
    correlation = df_clean['wse_ellips_m'].corr(df_clean['usgs_value_m_median'])
    ax1.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    # Plot 2: Demeaned comparison (single axis)
    gnss_demeaned = df_clean['wse_ellips_m'] - df_clean['wse_ellips_m'].mean()
    usgs_demeaned = df_clean['usgs_value_m_median'] - df_clean['usgs_value_m_median'].mean()
    
    ax2.plot(df_clean.index, gnss_demeaned, 
             color=colors['gnss'], linewidth=2.5, 
             marker='o', markersize=4, alpha=0.8,
             label='GNSS-IR WSE (demeaned)')
    
    ax2.plot(df_clean.index, usgs_demeaned, 
             color=colors['usgs'], linewidth=2.5,
             marker='s', markersize=3, alpha=0.8,
             label='USGS WL (demeaned)')
    
    # Styling for bottom plot
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Demeaned Water Level (m)', fontweight='bold')
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    ax2.legend(loc='upper left', framealpha=0.9)
    
    # Format dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add demeaned correlation
    demeaned_corr = gnss_demeaned.corr(usgs_demeaned)
    ax2.text(0.02, 0.98, f'Demeaned Correlation: {demeaned_corr:.3f}', 
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Enhanced comparison plot saved: {output_path}")
    return correlation, demeaned_corr

def create_quality_diagnostic_plot(df, station_name, year, output_path):
    """
    Create diagnostic plot to understand data quality issues
    
    Args:
        df: DataFrame with combined GNSS-IR and USGS data
        station_name: Station name
        year: Year for the data
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle(f'{station_name} Data Quality Diagnostics ({year})', 
                 fontsize=16, fontweight='bold')
    
    # Ensure we have datetime index
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
    
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    
    # 1. Data availability heatmap by month
    availability = df_copy.groupby('month').agg({
        'wse_ellips_m': lambda x: x.notna().sum(),
        'usgs_value_m_median': lambda x: x.notna().sum(),
    })
    
    # Add RH count if available
    if 'rh_count' in df_copy.columns:
        availability['rh_count'] = df_copy.groupby('month')['rh_count'].mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    axes[0,0].bar(months, availability['wse_ellips_m'], 
                  alpha=0.7, label='GNSS-IR', color=PLOT_COLORS['gnss'])
    axes[0,0].bar(months, availability['usgs_value_m_median'], 
                  alpha=0.7, label='USGS', color=PLOT_COLORS['usgs'])
    axes[0,0].set_title('Data Availability by Month')
    axes[0,0].set_ylabel('Days with Data')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Monthly correlation analysis
    monthly_corr = []
    for month in range(1, 13):
        month_data = df_copy[df_copy['month'] == month]
        if len(month_data) > 5:
            corr = month_data['wse_ellips_m'].corr(month_data['usgs_value_m_median'])
            monthly_corr.append(corr if not np.isnan(corr) else 0)
        else:
            monthly_corr.append(0)
    
    bars = axes[0,1].bar(months, monthly_corr, color=PLOT_COLORS['correlation'])
    axes[0,1].set_title('Monthly Correlation')
    axes[0,1].set_ylabel('Correlation Coefficient')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Color bars based on correlation quality
    for i, (bar, corr) in enumerate(zip(bars, monthly_corr)):
        if corr > 0.7:
            bar.set_color(PLOT_COLORS['quality_good'])
        elif corr > 0.3:
            bar.set_color(PLOT_COLORS['quality_medium'])
        else:
            bar.set_color(PLOT_COLORS['quality_poor'])
    
    # 3. Scatter plot with seasonal coloring
    valid_data = df.dropna(subset=['wse_ellips_m', 'usgs_value_m_median'])
    if len(valid_data) > 0:
        # Ensure valid_data has datetime index
        if not isinstance(valid_data.index, pd.DatetimeIndex):
            valid_data.index = pd.to_datetime(valid_data.index)
            
        seasons = valid_data.index.map(lambda x: 
            'Winter' if x.month in [12,1,2] else
            'Spring' if x.month in [3,4,5] else  
            'Summer' if x.month in [6,7,8] else 'Fall')
        
        season_colors = {'Winter': '#1f77b4', 'Spring': '#ff7f0e', 
                        'Summer': '#2ca02c', 'Fall': '#d62728'}
        
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            mask = seasons == season
            if mask.any():
                axes[1,0].scatter(valid_data.loc[mask, 'wse_ellips_m'], 
                                valid_data.loc[mask, 'usgs_value_m_median'],
                                c=season_colors[season], label=season, alpha=0.7, s=50)
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data['wse_ellips_m'], valid_data['usgs_value_m_median'])
        
        x_line = np.linspace(valid_data['wse_ellips_m'].min(), 
                           valid_data['wse_ellips_m'].max(), 100)
        y_line = slope * x_line + intercept
        axes[1,0].plot(x_line, y_line, 'k--', alpha=0.8, linewidth=2)
        
        axes[1,0].set_xlabel('GNSS-IR WSE (m)')
        axes[1,0].set_ylabel('USGS Water Level (m)')
        axes[1,0].set_title(f'Seasonal Scatter (R² = {r_value**2:.3f})')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Quality metrics vs residuals
    if 'rh_std_m' in df.columns and len(valid_data) > 0:
        residuals = valid_data['wse_ellips_m'] - valid_data['usgs_value_m_median']
        quality_data = df.loc[valid_data.index, 'rh_std_m'].dropna()
        
        if len(quality_data) > 0:
            axes[1,1].scatter(quality_data, residuals.loc[quality_data.index], 
                            alpha=0.6, s=50, color=PLOT_COLORS['correlation'])
            axes[1,1].set_xlabel('RH Standard Deviation (m)')
            axes[1,1].set_ylabel('WSE - USGS Residual (m)')
            axes[1,1].set_title('Data Quality vs Residuals')
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'RH Std data\nnot available', 
                           ha='center', va='center', transform=axes[1,1].transAxes,
                           fontsize=14)
    else:
        axes[1,1].text(0.5, 0.5, 'Quality metrics\nnot available', 
                       ha='center', va='center', transform=axes[1,1].transAxes,
                       fontsize=14)
        axes[1,1].set_title('Quality Metrics Not Available')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Quality diagnostic plot saved: {output_path}")

def investigate_seasonal_correlation_issues(df, station_name, year):
    """
    Comprehensive analysis of why seasonal correlations vary
    
    Args:
        df: DataFrame with combined GNSS-IR and USGS data
        station_name: Station name
        year: Year for analysis
        
    Returns:
        Dictionary with seasonal statistics
    """
    print(f"\n{'='*60}")
    print(f"SEASONAL CORRELATION INVESTIGATION: {station_name} {year}")
    print(f"{'='*60}")
    
    # Define seasons
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df_analysis = df.copy()
    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        df_analysis.index = pd.to_datetime(df_analysis.index)
    
    df_analysis['season'] = df_analysis.index.map(get_season)
    
    # Analyze each season
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_stats = {}
    
    for season in seasons:
        season_data = df_analysis[df_analysis['season'] == season]
        valid_data = season_data.dropna(subset=['wse_ellips_m', 'usgs_value_m_median'])
        
        if len(valid_data) > 3:
            correlation = valid_data['wse_ellips_m'].corr(valid_data['usgs_value_m_median'])
            
            # Calculate statistics
            gnss_mean = valid_data['wse_ellips_m'].mean()
            gnss_std = valid_data['wse_ellips_m'].std()
            usgs_mean = valid_data['usgs_value_m_median'].mean()
            usgs_std = valid_data['usgs_value_m_median'].std()
            
            # Data quality metrics
            rh_count_mean = valid_data.get('rh_count', pd.Series()).mean()
            rh_std_mean = valid_data.get('rh_std_m', pd.Series()).mean()
            
            seasonal_stats[season] = {
                'correlation': correlation,
                'n_points': len(valid_data),
                'gnss_mean': gnss_mean,
                'gnss_std': gnss_std,
                'usgs_mean': usgs_mean,
                'usgs_std': usgs_std,
                'rh_count_mean': rh_count_mean,
                'rh_std_mean': rh_std_mean,
                'date_range': f"{valid_data.index.min().strftime('%m/%d')} - {valid_data.index.max().strftime('%m/%d')}"
            }
            
            print(f"\n{season.upper()} ANALYSIS:")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  Data points: {len(valid_data)}")
            print(f"  Date range: {seasonal_stats[season]['date_range']}")
            print(f"  GNSS-IR: mean={gnss_mean:.3f}m, std={gnss_std:.3f}m")
            print(f"  USGS: mean={usgs_mean:.3f}m, std={usgs_std:.3f}m")
            if not pd.isna(rh_count_mean):
                print(f"  Avg RH retrievals/day: {rh_count_mean:.1f}")
            if not pd.isna(rh_std_mean):
                print(f"  Avg RH std: {rh_std_mean:.3f}m")
        else:
            print(f"\n{season.upper()}: Insufficient data ({len(valid_data)} points)")
    
    return seasonal_stats

def detect_outliers_and_anomalies(df, station_name, year):
    """
    Detect outliers that might be affecting correlations
    
    Args:
        df: DataFrame with combined data
        station_name: Station name
        year: Year for analysis
        
    Returns:
        Tuple of (outliers_df, clean_correlation)
    """
    print(f"\n{'='*50}")
    print(f"OUTLIER DETECTION ANALYSIS")
    print(f"{'='*50}")
    
    valid_data = df.dropna(subset=['wse_ellips_m', 'usgs_value_m_median'])
    
    if len(valid_data) < 10:
        print("Insufficient data for outlier analysis")
        return pd.DataFrame(), None
    
    # Calculate residuals
    residuals = valid_data['wse_ellips_m'] - valid_data['usgs_value_m_median']
    
    # Outlier detection using IQR method
    Q1 = residuals.quantile(0.25)
    Q3 = residuals.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = valid_data[(residuals < lower_bound) | (residuals > upper_bound)]
    
    print(f"Total data points: {len(valid_data)}")
    print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(valid_data)*100:.1f}%)")
    print(f"Residual statistics:")
    print(f"  Mean: {residuals.mean():.3f}m")
    print(f"  Std: {residuals.std():.3f}m")
    print(f"  IQR bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
    
    if len(outliers) > 0:
        print(f"\nOutlier dates and residuals:")
        for date, row in outliers.iterrows():
            residual = residuals.loc[date]
            print(f"  {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}: {residual:.3f}m (GNSS: {row['wse_ellips_m']:.3f}, USGS: {row['usgs_value_m_median']:.3f})")
    
    # Correlation without outliers
    clean_data = valid_data[(residuals >= lower_bound) & (residuals <= upper_bound)]
    clean_correlation = clean_data['wse_ellips_m'].corr(clean_data['usgs_value_m_median'])
    original_correlation = valid_data['wse_ellips_m'].corr(valid_data['usgs_value_m_median'])
    
    print(f"\nCorrelation analysis:")
    print(f"  Original correlation: {original_correlation:.4f}")
    print(f"  Without outliers: {clean_correlation:.4f}")
    print(f"  Improvement: {clean_correlation - original_correlation:.4f}")
    
    return outliers, clean_correlation

def create_spring_investigation_plot(df, station_name, year, output_path):
    """
    Create detailed plot investigating Spring correlation issues
    
    Args:
        df: DataFrame with combined data
        station_name: Station name
        year: Year
        output_path: Path to save plot
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Filter to Spring months (March, April, May)
    spring_data = df[df.index.month.isin([3, 4, 5])]
    valid_spring = spring_data.dropna(subset=['wse_ellips_m', 'usgs_value_m_median'])
    
    if len(valid_spring) == 0:
        print("No valid Spring data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=150)
    fig.suptitle(f'{station_name} Spring Data Investigation ({year})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Time series comparison
    axes[0,0].plot(valid_spring.index, valid_spring['wse_ellips_m'], 
                   'b-o', label='GNSS-IR WSE', markersize=6, color=PLOT_COLORS['gnss'])
    ax_twin = axes[0,0].twinx()
    ax_twin.plot(valid_spring.index, valid_spring['usgs_value_m_median'], 
                 'r-s', label='USGS WL', markersize=4, color=PLOT_COLORS['usgs'])
    
    axes[0,0].set_title('Spring Time Series Comparison')
    axes[0,0].set_ylabel('GNSS-IR WSE (m)', color=PLOT_COLORS['gnss'])
    ax_twin.set_ylabel('USGS WL (m)', color=PLOT_COLORS['usgs'])
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Scatter plot
    axes[0,1].scatter(valid_spring['wse_ellips_m'], valid_spring['usgs_value_m_median'], 
                      alpha=0.7, s=80, color=PLOT_COLORS['correlation'])
    
    # Add regression line
    if len(valid_spring) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_spring['wse_ellips_m'], valid_spring['usgs_value_m_median'])
        
        x_line = np.linspace(valid_spring['wse_ellips_m'].min(), 
                           valid_spring['wse_ellips_m'].max(), 100)
        y_line = slope * x_line + intercept
        axes[0,1].plot(x_line, y_line, 'r--', alpha=0.8)
        
        axes[0,1].set_title(f'Spring Scatter (R² = {r_value**2:.3f})')
    else:
        axes[0,1].set_title('Spring Scatter (Insufficient Data)')
    
    axes[0,1].set_xlabel('GNSS-IR WSE (m)')
    axes[0,1].set_ylabel('USGS WL (m)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Residuals over time
    residuals = valid_spring['wse_ellips_m'] - valid_spring['usgs_value_m_median']
    axes[1,0].plot(valid_spring.index, residuals, 'go-', markersize=6, color=PLOT_COLORS['correlation'])
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Spring Residuals (GNSS-IR - USGS)')
    axes[1,0].set_ylabel('Residual (m)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Data quality metrics
    if 'rh_count' in valid_spring.columns:
        axes[1,1].scatter(valid_spring['rh_count'], residuals, 
                         alpha=0.7, s=80, color=PLOT_COLORS['correlation'])
        axes[1,1].set_xlabel('RH Retrievals per Day')
        axes[1,1].set_ylabel('Residual (m)')
        axes[1,1].set_title('Residuals vs Data Quality')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Quality metrics\nnot available', 
                       ha='center', va='center', transform=axes[1,1].transAxes,
                       fontsize=14)
        axes[1,1].set_title('Quality Metrics Not Available')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Spring investigation plot saved: {output_path}")

def run_comprehensive_correlation_investigation(df, station_name, year, output_dir):
    """
    Run all correlation investigation tools
    
    Args:
        df: DataFrame with combined data
        station_name: Station name
        year: Year
        output_dir: Output directory
        
    Returns:
        Tuple of (seasonal_stats, outliers, clean_correlation)
    """
    import os
    output_dir = Path(output_dir)
    
    print(f"Starting comprehensive correlation investigation for {station_name} {year}")
    
    # 1. Seasonal analysis
    seasonal_stats = investigate_seasonal_correlation_issues(df, station_name, year)
    
    # 2. Outlier detection
    outliers, clean_correlation = detect_outliers_and_anomalies(df, station_name, year)
    
    # 3. Create comparison plot
    comparison_path = output_dir / f'{station_name}_{year}_comparison.png'
    corr, demeaned_corr = create_comparison_plot(df, station_name, year, comparison_path)
    
    # 4. Create quality diagnostics
    diagnostic_path = output_dir / f'{station_name}_{year}_quality_diagnostics.png'
    create_quality_diagnostic_plot(df, station_name, year, diagnostic_path)
    
    # 5. Create Spring investigation plot
    spring_plot_path = output_dir / f'{station_name}_{year}_spring_investigation.png'
    create_spring_investigation_plot(df, station_name, year, spring_plot_path)
    
    # 6. Summary recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR IMPROVEMENT:")
    print(f"{'='*60}")
    
    if 'Spring' in seasonal_stats:
        spring_corr = seasonal_stats['Spring']['correlation']
        if spring_corr < 0.3:
            print("CRITICAL: Spring correlation is critically low (<0.3)")
            print("   - Check for systematic bias or environmental factors")
            print("   - Consider vegetation effects on GNSS-IR signals")
            print("   - Verify USGS gauge is representative of GNSS location")
        elif spring_corr < 0.5:
            print("MODERATE: Spring correlation is moderate (0.3-0.5)")
            print("   - Consider data quality filtering")
            print("   - Check for temporal alignment issues")
    
    if len(outliers) > len(df) * 0.1:  # More than 10% outliers
        print("WARNING: High outlier percentage detected")
        print("   - Implement outlier filtering in processing pipeline")
        print("   - Investigate dates with extreme residuals")
    
    if clean_correlation and corr and (clean_correlation - corr) > 0.1:
        print("SUCCESS: Outlier removal significantly improves correlation")
        print("   - Consider implementing automated outlier filtering")
    
    print(f"\nGenerated plots:")
    print(f"  Enhanced comparison: {comparison_path}")
    print(f"  Quality diagnostics: {diagnostic_path}")
    print(f"  Spring investigation: {spring_plot_path}")
    
    return seasonal_stats, outliers, clean_correlation
