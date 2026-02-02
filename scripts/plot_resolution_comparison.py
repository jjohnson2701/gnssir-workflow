# ABOUTME: Generates correlation vs temporal resolution comparison plots.
# ABOUTME: Shows how data aggregation affects correlation and RMSE for GNSS-IR measurements.

"""Generate resolution comparison plot showing correlation and RMSE vs temporal aggregation."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def aggregate_to_resolution(df, gnss_col, ref_col, time_col, resolution):
    """
    Aggregate data to specified temporal resolution.

    Args:
        df: DataFrame with matched data
        gnss_col: GNSS demeaned column name
        ref_col: Reference demeaned column name
        time_col: Datetime column name
        resolution: Pandas offset string (e.g., 'h', '3h', '6h', 'D', 'W')

    Returns:
        Aggregated DataFrame with median values
    """
    df_copy = df.copy().set_index(time_col)

    agg_df = df_copy[[gnss_col, ref_col]].resample(resolution).median().dropna()
    agg_df = agg_df.reset_index()
    agg_df.columns = ['datetime', 'gnss_dm', 'ref_dm']

    return agg_df


def plot_resolution_comparison(
    matched_csv_path: Path,
    station_name: str,
    year: int,
    output_path: Path,
    ref_source: str = "USGS"
) -> Path:
    """
    Create resolution comparison plot.

    Args:
        matched_csv_path: Path to matched subdaily CSV
        station_name: Station name
        year: Year of data
        output_path: Output path for plot
        ref_source: Reference source type

    Returns:
        Path to generated plot
    """
    # Load data
    df = pd.read_csv(matched_csv_path)
    df['gnss_datetime'] = pd.to_datetime(df['gnss_datetime'], format='mixed', utc=True)

    # Identify columns
    if 'gnss_wse_dm' in df.columns:
        gnss_dm_col = 'gnss_wse_dm'
    elif 'gnss_dm' in df.columns:
        gnss_dm_col = 'gnss_dm'
    else:
        raise ValueError(f"No GNSS demeaned column found")

    # Find reference demeaned column (various naming conventions)
    ref_dm_candidates = ['usgs_wl_dm', 'coops_dm', 'bartlett_dm', 'bartlett_cove_dm']
    ref_dm_col = None
    for col in ref_dm_candidates:
        if col in df.columns:
            ref_dm_col = col
            break
    # Also check for any column ending with '_dm' that's not gnss
    if ref_dm_col is None:
        for col in df.columns:
            if col.endswith('_dm') and 'gnss' not in col.lower():
                ref_dm_col = col
                break
    if ref_dm_col is None:
        raise ValueError(f"No reference demeaned column found. Available: {list(df.columns)}")

    print(f"Loaded {len(df)} points, using {gnss_dm_col} vs {ref_dm_col}")

    # Calculate statistics at different resolutions
    resolutions = [
        ('Individual\n(~min)', None),
        ('6-hourly\nmedian', '6h'),
        ('Daily\nmedian', 'D'),
        ('Weekly\nmedian', 'W')
    ]

    stats = []
    for res_name, res_code in resolutions:
        if res_code is None:
            # Individual points
            corr = df[gnss_dm_col].corr(df[ref_dm_col])
            rmse = np.sqrt(np.mean((df[gnss_dm_col] - df[ref_dm_col])**2))
            n_points = len(df)
            data = df[[gnss_dm_col, ref_dm_col]].copy()
            data.columns = ['gnss_dm', 'ref_dm']
        else:
            # Aggregated
            agg_df = aggregate_to_resolution(df, gnss_dm_col, ref_dm_col, 'gnss_datetime', res_code)
            corr = agg_df['gnss_dm'].corr(agg_df['ref_dm'])
            rmse = np.sqrt(np.mean((agg_df['gnss_dm'] - agg_df['ref_dm'])**2))
            n_points = len(agg_df)
            data = agg_df

        stats.append({
            'name': res_name,
            'corr': corr,
            'rmse': rmse,
            'n': n_points,
            'data': data
        })

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, height_ratios=[1, 1.2, 1.2])

    # Top row: Bar charts
    ax_corr = fig.add_subplot(gs[0, :3])
    ax_rmse = fig.add_subplot(gs[0, 3])

    # Correlation bars
    names = [s['name'] for s in stats]
    corrs = [s['corr'] for s in stats]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stats)))

    bars = ax_corr.bar(names, corrs, color=colors, edgecolor='black', linewidth=0.5)

    # Dynamic y-axis based on actual correlation range - with padding for labels
    min_corr = min(corrs)
    max_corr = max(corrs)
    y_range = max(max_corr - min_corr, 0.05)  # Ensure minimum range of 0.05
    y_min = max(0, min_corr - 0.2 * y_range)  # 20% padding below
    y_max = min(1.0, max_corr + 0.3 * y_range)  # 30% padding above for labels

    # Add text labels with offset relative to y-range
    text_offset = 0.05 * (y_max - y_min)
    for i, (bar, s) in enumerate(zip(bars, stats)):
        height = bar.get_height()
        ax_corr.text(bar.get_x() + bar.get_width()/2, height + text_offset,
                    f'{s["corr"]:.3f}', ha='center', va='bottom', fontsize=9)

    ax_corr.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='r=0.9')
    ax_corr.set_ylabel('Correlation (r)', fontsize=12)
    ax_corr.set_title('Correlation vs Temporal Resolution (Full Year)', fontsize=12, fontweight='bold')
    ax_corr.set_ylim(y_min, y_max)
    ax_corr.grid(axis='y', alpha=0.3)

    # RMSE bars with padding for labels
    rmses = [s['rmse'] * 100 for s in stats]  # Convert to cm
    bars_rmse = ax_rmse.bar(range(len(stats)), rmses, color=colors, edgecolor='black', linewidth=0.5)
    for i, (bar, rmse) in enumerate(zip(bars_rmse, rmses)):
        height = bar.get_height()
        ax_rmse.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                    f'{rmse:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

    # Add padding above for labels
    max_rmse = max(rmses)
    ax_rmse.set_ylim(0, max_rmse * 1.15)  # 15% padding above

    ax_rmse.set_ylabel('RMSE (cm)', fontsize=12)
    ax_rmse.set_title('RMSE', fontsize=12, fontweight='bold')
    ax_rmse.set_xticks(range(len(stats)))
    ax_rmse.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax_rmse.grid(axis='y', alpha=0.3)

    # Bottom rows: Quarterly time series in 2x2 grid
    df_sorted = df.sort_values('gnss_datetime')
    quarters = [
        (1, 3, 'Q1: Jan-Mar'),
        (4, 6, 'Q2: Apr-Jun'),
        (7, 9, 'Q3: Jul-Sep'),
        (10, 12, 'Q4: Oct-Dec')
    ]

    # Map quarters to 2x2 grid positions
    grid_positions = [
        (1, slice(0, 2)),  # Q1: row 1, cols 0-1
        (1, slice(2, 4)),  # Q2: row 1, cols 2-3
        (2, slice(0, 2)),  # Q3: row 2, cols 0-1
        (2, slice(2, 4))   # Q4: row 2, cols 2-3
    ]

    for i, (start_month, end_month, title) in enumerate(quarters):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        q_data = df_sorted[
            (df_sorted['gnss_datetime'].dt.month >= start_month) &
            (df_sorted['gnss_datetime'].dt.month <= end_month)
        ].copy()

        if len(q_data) > 0:
            # Calculate quarterly stats
            q_corr = q_data[gnss_dm_col].corr(q_data[ref_dm_col])
            q_n = len(q_data)

            # Calculate fractional day of year to preserve intra-day timing
            q_data['doy_frac'] = (q_data['gnss_datetime'].dt.dayofyear +
                                  q_data['gnss_datetime'].dt.hour / 24.0 +
                                  q_data['gnss_datetime'].dt.minute / 1440.0)

            ax.plot(q_data['doy_frac'], q_data[gnss_dm_col], 'o-',
                   markersize=2, linewidth=0.5, alpha=0.6, label='GNSS-IR', color='#2471A3')
            ax.plot(q_data['doy_frac'], q_data[ref_dm_col], '-',
                   linewidth=1.5, alpha=0.8, label=ref_source, color='#C0392B')

            ax.set_ylabel('Demeaned (m)', fontsize=9)
            ax.set_xlabel('Day of Year', fontsize=9)
            ax.set_title(f'{title}\n(r={q_corr:.2f}, N={q_n:,})', fontsize=9)
            ax.legend(fontsize=7, loc='best')
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)

    # Main title
    fig.suptitle(
        f'{station_name} {year}: Correlation vs Temporal Resolution\n'
        f'Full year: N={len(df):,} individual readings',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Saved resolution comparison to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate resolution comparison plot')
    parser.add_argument('--station', type=str, required=True, help='Station name')
    parser.add_argument('--year', type=int, default=2024, help='Year')
    args = parser.parse_args()

    # Set paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results_annual' / args.station
    matched_csv = results_dir / f'{args.station}_{args.year}_subdaily_matched.csv'
    output_path = results_dir / f'{args.station}_{args.year}_resolution_comparison.png'

    if not matched_csv.exists():
        print(f"Error: {matched_csv} not found")
        return 1

    # Determine reference source
    station_sources = {
        'MDAI': 'USGS',
        'FORA': 'USGS',
        'VALR': 'CO-OPS',
        'GLBX': 'CO-OPS'
    }
    ref_source = station_sources.get(args.station, 'Reference')

    plot_resolution_comparison(
        matched_csv_path=matched_csv,
        station_name=args.station,
        year=args.year,
        output_path=output_path,
        ref_source=ref_source
    )

    return 0


if __name__ == '__main__':
    exit(main())
