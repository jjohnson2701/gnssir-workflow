# ABOUTME: Results aggregator for daily GNSS-IR reflector height files
# ABOUTME: Combines daily outputs into annual datasets with enriched per-arc and daily statistics

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Canonical gnssir v3 output column names (17 columns)
# Based on gnssrefl v3.10.0 output format
GNSSIR_V3_COLUMNS = [
    "year",  # (1) Year
    "doy",  # (2) Day of year
    "RH",  # (3) Reflector height in meters
    "sat",  # (4) Satellite number
    "UTCtime",  # (5) UTC time in hours
    "Azim",  # (6) Azimuth in degrees
    "Amp",  # (7) Amplitude (v/v)
    "eminO",  # (8) Minimum elevation angle in degrees
    "emaxO",  # (9) Maximum elevation angle in degrees
    "NumbOf",  # (10) Number of values used
    "freq",  # (11) Frequency code
    "rise",  # (12) Rising (1) or setting (-1)
    "EdotF",  # (13) Edot/F value in hours
    "PkNoise",  # (14) Peak to noise ratio
    "DelT",  # (15) Delta T in minutes
    "MJD",  # (16) Modified Julian Date
    "refr_model",  # (17) Refraction model (0 is none)
]

# Map gnssrefl frequency codes to carrier-frequency bands.
# Grouped by actual RF band (not constellation) so interfrequency
# comparisons measure wavelength-dependent surface interaction.
FREQ_CODE_TO_GROUP = {
    1: "L1",       # GPS L1 C/A (1575.42 MHz, 0.190 m)
    2: "L2C",      # GPS L2 P(Y) (1227.60 MHz)
    20: "L2C",     # GPS L2C    (1227.60 MHz, 0.244 m)
    5: "L5",       # GPS L5     (1176.45 MHz, 0.255 m)
    101: "L1",     # GLONASS L1 (~1602 MHz)
    102: "L2C",    # GLONASS L2 (~1246 MHz)
    105: "L5",     # GLONASS L3 (1202.025 MHz)
    201: "L1",     # Galileo E1 (1575.42 MHz)
    205: "L5",     # Galileo E5a (1176.45 MHz)
    206: "L5",     # Galileo E5b (1207.14 MHz)
    207: "L5",     # Galileo E5  (AltBOC composite)
    208: "E6",     # Galileo E6 (1278.75 MHz, unique)
    301: "L1",     # BeiDou B1C (1575.42 MHz)
    302: "L1",     # BeiDou B1  (1561.098 MHz)
    306: "B3",     # BeiDou B3  (1268.52 MHz, unique)
}

# Azimuth bin size in degrees — 10-degree bins provide the spatial
# resolution needed to identify land contamination and directional
# ice signatures (90-degree bins hide critical structure).
AZ_BIN_SIZE = 10


def _freq_group(code):
    """Map a gnssrefl frequency code to its RF band group."""
    return FREQ_CODE_TO_GROUP.get(int(code), "OTHER")


def _az_bin(azimuth):
    """Map azimuth (0-360) to a 10-degree bin. Returns the bin lower bound."""
    return int(azimuth // AZ_BIN_SIZE) * AZ_BIN_SIZE


def add_derived_columns(df, antenna_height_m=None):
    """Add derived columns to a per-arc DataFrame.

    Adds: azimuth_bin (10-degree), freq_group, and optionally wse.
    Operates in-place for efficiency, returns the same DataFrame.
    """
    if "freq" in df.columns:
        df["freq_group"] = df["freq"].map(_freq_group).fillna("OTHER")
    if "Azim" in df.columns:
        df["azimuth_bin"] = df["Azim"].apply(_az_bin).astype(np.int16)
    if antenna_height_m is not None and "RH" in df.columns:
        df["wse"] = antenna_height_m - df["RH"]
    return df


def compute_enriched_daily(df):
    """Compute daily statistics grouped by date x azimuth_bin x freq_group.

    Returns a DataFrame with RH, amplitude, and peak2noise statistics
    per group, plus pooled rows (azimuth_bin=-1, freq_group='ALL').
    """
    required = {"date", "RH", "azimuth_bin", "freq_group"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logging.error(f"Cannot compute enriched daily: missing columns {missing}")
        return None

    has_amp = "Amp" in df.columns
    has_p2n = "PkNoise" in df.columns
    has_wse = "wse" in df.columns

    def _agg_group(g):
        """Aggregate a single group to a stats row."""
        row = {
            "rh_count": len(g),
            "rh_mean": g["RH"].mean(),
            "rh_median": g["RH"].median(),
            "rh_std": g["RH"].std(),
            "rh_min": g["RH"].min(),
            "rh_max": g["RH"].max(),
            "rh_range": g["RH"].max() - g["RH"].min(),
        }
        if has_amp:
            amp_mean = g["Amp"].mean()
            amp_std = g["Amp"].std()
            row["amp_mean"] = amp_mean
            row["amp_std"] = amp_std
            row["amp_cv"] = amp_std / amp_mean if amp_mean > 0 else np.nan
            row["amp_min"] = g["Amp"].min()
            row["amp_max"] = g["Amp"].max()
        if has_p2n:
            row["p2n_mean"] = g["PkNoise"].mean()
            row["p2n_std"] = g["PkNoise"].std()
            row["p2n_min"] = g["PkNoise"].min()
            row["p2n_max"] = g["PkNoise"].max()
        if has_wse:
            row["wse_mean"] = g["wse"].mean()
            row["wse_std"] = g["wse"].std()
        return pd.Series(row)

    # Per azimuth_bin x freq_group
    grouped = df.groupby(["date", "azimuth_bin", "freq_group"]).apply(
        _agg_group, include_groups=False
    ).reset_index()

    # Pooled across all azimuths and frequencies (backward-compatible daily row)
    pooled = df.groupby("date").apply(_agg_group, include_groups=False).reset_index()
    pooled["azimuth_bin"] = np.int8(-1)
    pooled["freq_group"] = "ALL"

    # Pooled per azimuth (all freqs)
    pooled_az = df.groupby(["date", "azimuth_bin"]).apply(
        _agg_group, include_groups=False
    ).reset_index()
    pooled_az["freq_group"] = "ALL"

    # Pooled per freq (all azimuths)
    pooled_freq = df.groupby(["date", "freq_group"]).apply(
        _agg_group, include_groups=False
    ).reset_index()
    pooled_freq["azimuth_bin"] = np.int8(-1)

    enriched = pd.concat([grouped, pooled, pooled_az, pooled_freq], ignore_index=True)
    enriched.sort_values(["date", "azimuth_bin", "freq_group"], inplace=True)
    return enriched


def compute_interfreq_daily(df):
    """Compute daily interfrequency divergence per azimuth bin.

    For each date x azimuth_bin, computes mean RH and amplitude per
    frequency band and the pairwise differences (L1-L2C, L1-L5).
    Divergence indicates surface penetration (ice, snow, vegetation).

    Returns DataFrame or None if insufficient frequency diversity.
    """
    required = {"date", "RH", "Amp", "azimuth_bin", "freq_group"}
    if not required.issubset(df.columns):
        return None

    # Only include the main bands for interfreq comparison
    main_bands = ["L1", "L2C", "L5"]
    df_main = df[df["freq_group"].isin(main_bands)]
    if df_main.empty:
        return None

    # Per date x azimuth_bin x freq_group means
    grp = df_main.groupby(["date", "azimuth_bin", "freq_group"]).agg(
        rh_mean=("RH", "mean"),
        amp_mean=("Amp", "mean"),
        arc_count=("RH", "count"),
    ).reset_index()

    # Pivot so each band becomes a column
    rows = []
    for (date, az_bin), sub in grp.groupby(["date", "azimuth_bin"]):
        row = {"date": date, "azimuth_bin": az_bin}
        for _, r in sub.iterrows():
            fg = r["freq_group"]
            row[f"{fg}_rh_mean"] = r["rh_mean"]
            row[f"{fg}_amp_mean"] = r["amp_mean"]
            row[f"{fg}_count"] = int(r["arc_count"])

        # Pairwise RH differences
        l1 = row.get("L1_rh_mean")
        l2c = row.get("L2C_rh_mean")
        l5 = row.get("L5_rh_mean")
        if l1 is not None and l2c is not None:
            row["L1_L2C_rh_diff"] = l1 - l2c
        if l1 is not None and l5 is not None:
            row["L1_L5_rh_diff"] = l1 - l5

        # Pairwise amplitude differences
        l1a = row.get("L1_amp_mean")
        l2ca = row.get("L2C_amp_mean")
        l5a = row.get("L5_amp_mean")
        if l1a is not None and l2ca is not None:
            row["L1_L2C_amp_diff"] = l1a - l2ca
        if l1a is not None and l5a is not None:
            row["L1_L5_amp_diff"] = l1a - l5a

        rows.append(row)

    if not rows:
        return None

    result = pd.DataFrame(rows)
    result.sort_values(["date", "azimuth_bin"], inplace=True)
    return result


def combine_daily_rh_results(
    station_id_4char_lower,
    year,
    daily_rh_base_dir,
    annual_results_dir,
    antenna_height_m=None,
):
    """
    Combine daily reflector height results into annual datasets.

    Produces three outputs:
      1. {STATION}_{year}_combined_rh.csv    -- backward-compatible daily RH stats
      2. {STATION}_{year}_per_arc.parquet     -- all per-arc data with derived columns
      3. {STATION}_{year}_daily_enriched.parquet -- daily stats by azimuth_bin x freq_group

    Args:
        station_id_4char_lower (str): Station ID in 4-character lowercase
        year (int): Year (4 digits)
        daily_rh_base_dir (str or Path): Base directory containing daily RH text files
        annual_results_dir (str or Path): Output directory for combined outputs
        antenna_height_m (float, optional): Antenna height for WSE computation

    Returns:
        Path or None: Path to the combined CSV file on success, None on failure
    """
    daily_rh_base_dir = Path(daily_rh_base_dir)
    annual_results_dir = Path(annual_results_dir)

    # Ensure the output directory exists
    annual_results_dir.mkdir(parents=True, exist_ok=True)

    # Define output CSV path
    output_csv_path = (
        annual_results_dir / f"{station_id_4char_lower.upper()}_{year}_combined_rh.csv"
    )

    try:
        # Find all daily RH files - they should be named station_year_doy.txt
        file_pattern = f"{station_id_4char_lower}_{year}_*.txt"
        rh_files = sorted(daily_rh_base_dir.glob(file_pattern))

        if not rh_files:
            logging.warning(
                f"No RH files found matching pattern {file_pattern} in {daily_rh_base_dir}"
            )
            return None

        logging.info(f"Found {len(rh_files)} RH files to combine")
        logging.info(f"Files: {[f.name for f in rh_files]}")

        # Initialize a list to store DataFrames
        all_data = []
        processing_errors = []

        # Process each RH file
        for rh_file in rh_files:
            try:
                # Check if file format has comment lines at top (indicated by % symbol)
                with open(rh_file, "r") as f:
                    lines = f.readlines()

                # Skip empty files
                if not lines:
                    logging.warning(f"Empty file: {rh_file.name}, skipping")
                    processing_errors.append(f"{rh_file.name}: Empty file")
                    continue

                # Count the number of header lines (starting with %)
                header_lines = 0
                for line in lines:
                    if line.strip().startswith("%"):
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
                column_descriptions = []

                # Look for the line with column descriptions (typically line 3)
                if header_lines >= 3:
                    # Line 3 (index 2) typically has descriptions like "% year, doy, RH..."
                    desc_line = lines[2]
                    if desc_line.startswith("%"):
                        # Remove the % and split by commas
                        desc_parts = desc_line[1:].strip().split(",")
                        column_descriptions = [part.strip() for part in desc_parts]
                        logging.debug(f"Column descriptions: {column_descriptions}")

                # Now read the data, skipping the header lines
                try:
                    df = pd.read_csv(
                        rh_file,
                        skiprows=header_lines,
                        sep=r"\s+",  # Space-delimited (replaces deprecated delim_whitespace)
                        header=None,
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

                # Assign column names using canonical gnssir v3 format
                # The gnssir output has 17 space-delimited columns with a fixed format
                num_cols = len(df.columns)
                if num_cols == len(GNSSIR_V3_COLUMNS):
                    # Perfect match - use canonical names
                    df.columns = GNSSIR_V3_COLUMNS
                    logging.debug(
                        f"Assigned canonical gnssir v3 column names: {df.columns.tolist()}"
                    )
                elif num_cols < len(GNSSIR_V3_COLUMNS):
                    # Fewer columns than expected - use canonical names for available columns
                    df.columns = GNSSIR_V3_COLUMNS[:num_cols]
                    logging.warning(
                        f"File has {num_cols} columns, expected {len(GNSSIR_V3_COLUMNS)}. "
                        "Using partial canonical names."
                    )
                else:
                    # More columns than expected - use canonical names + generic for extras
                    col_names = GNSSIR_V3_COLUMNS + [
                        f"Col{i}" for i in range(num_cols - len(GNSSIR_V3_COLUMNS))
                    ]
                    df.columns = col_names
                    logging.warning(
                        f"File has {num_cols} columns, expected {len(GNSSIR_V3_COLUMNS)}. "
                        "Added generic names for extras."
                    )

                # Extract the DOY from the filename
                filename_parts = rh_file.stem.split("_")
                if len(filename_parts) >= 3:
                    doy = filename_parts[-1]
                    try:
                        doy_int = int(doy)
                        # Add DOY as a column if it doesn't exist
                        if "doy" not in df.columns:
                            df["doy"] = doy_int
                            logging.debug(f"Added DOY column with value {doy_int}")
                    except ValueError:
                        logging.warning(
                            f"Could not extract valid DOY from filename: {rh_file.name}"
                        )
                else:
                    logging.warning(
                        f"Filename format doesn't match expected pattern: {rh_file.name}"
                    )

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

        # Sort by DOY and time
        sort_cols = [c for c in ["doy", "UTCtime"] if c in combined_df.columns]
        if sort_cols:
            combined_df.sort_values(sort_cols, inplace=True)

        # Add date column
        if "year" in combined_df.columns and "doy" in combined_df.columns:
            try:
                combined_df["date"] = combined_df.apply(
                    lambda row: (
                        datetime(int(row["year"]), 1, 1) + timedelta(days=int(row["doy"]) - 1)
                    ).strftime("%Y-%m-%d"),
                    axis=1,
                )
            except Exception as e:
                logging.warning(f"Could not add date column: {e}")

        if "date" not in combined_df.columns:
            logging.error("Cannot perform aggregation: date column is missing")
            combined_df.to_csv(output_csv_path, index=False)
            return output_csv_path

        if "RH" not in combined_df.columns:
            logging.error("Cannot identify RH column for aggregation")
            combined_df.to_csv(output_csv_path, index=False)
            return output_csv_path

        # --- Add derived columns for enriched analysis ---
        add_derived_columns(combined_df, antenna_height_m=antenna_height_m)
        logging.info("Added derived columns: azimuth_bin, freq_group"
                      + (", wse" if antenna_height_m is not None else ""))

        # --- Save per-arc parquet (all columns, typed efficiently) ---
        try:
            per_arc_path = (
                annual_results_dir
                / f"{station_id_4char_lower.upper()}_{year}_per_arc.parquet"
            )
            combined_df.to_parquet(per_arc_path, index=False, engine="pyarrow")
            logging.info(
                f"Per-arc parquet saved to {per_arc_path} "
                f"({len(combined_df)} arcs, "
                f"{per_arc_path.stat().st_size / 1024:.0f} KB)"
            )
        except Exception as e:
            logging.warning(f"Could not write per-arc parquet: {e}")

        # --- Compute enriched daily stats (by azimuth_bin x freq_group) ---
        try:
            enriched = compute_enriched_daily(combined_df)
            if enriched is not None:
                station_upper = station_id_4char_lower.upper()
                # Primary output: combined_enriched.parquet
                enriched_path = (
                    annual_results_dir / f"{station_upper}_{year}_combined_enriched.parquet"
                )
                enriched.to_parquet(enriched_path, index=False, engine="pyarrow")
                # Backward-compatible alias for dashboard (reads daily_enriched.parquet)
                compat_path = (
                    annual_results_dir / f"{station_upper}_{year}_daily_enriched.parquet"
                )
                enriched.to_parquet(compat_path, index=False, engine="pyarrow")
                n_groups = len(enriched)
                n_days = enriched["date"].nunique()
                logging.info(
                    f"Enriched parquet saved to {enriched_path} "
                    f"({n_groups} rows across {n_days} days)"
                )
        except Exception as e:
            logging.warning(f"Could not compute/write enriched daily: {e}")

        # --- Compute interfrequency divergence ---
        try:
            interfreq = compute_interfreq_daily(combined_df)
            if interfreq is not None:
                interfreq_path = (
                    annual_results_dir
                    / f"{station_id_4char_lower.upper()}_{year}_combined_interfreq.parquet"
                )
                interfreq.to_parquet(interfreq_path, index=False, engine="pyarrow")
                logging.info(
                    f"Interfreq parquet saved to {interfreq_path} "
                    f"({len(interfreq)} rows across {interfreq['date'].nunique()} days)"
                )
        except Exception as e:
            logging.warning(f"Could not compute/write interfreq: {e}")

        # --- Backward-compatible daily CSV (pooled across all azimuths/freqs) ---
        try:
            daily_agg = combined_df.groupby("date").agg(
                {"RH": ["count", "mean", "median", "std", "min", "max"]}
            )
            daily_agg.columns = ["_".join(col).strip() for col in daily_agg.columns.values]
            daily_agg.rename(
                columns={
                    "RH_count": "rh_count",
                    "RH_mean": "rh_mean_m",
                    "RH_median": "rh_median_m",
                    "RH_std": "rh_std_m",
                    "RH_min": "rh_min_m",
                    "RH_max": "rh_max_m",
                },
                inplace=True,
            )
            daily_agg.reset_index(inplace=True)
            daily_agg["datetime"] = pd.to_datetime(daily_agg["date"])
            daily_agg["year"] = daily_agg["datetime"].dt.year
            daily_agg["doy"] = daily_agg["datetime"].dt.strftime("%j").astype(int)

            daily_agg.to_csv(output_csv_path, index=False)
            logging.info(
                f"Daily aggregation: {len(combined_df)} arcs -> "
                f"{len(daily_agg)} days saved to {output_csv_path}"
            )
        except Exception as e:
            logging.error(f"Error during daily CSV aggregation: {e}")
            combined_df.to_csv(output_csv_path, index=False)

        # --- Also keep combined_raw.csv for existing consumers ---
        try:
            raw_csv_path = (
                annual_results_dir
                / f"{station_id_4char_lower.upper()}_{year}_combined_raw.csv"
            )
            combined_df.to_csv(raw_csv_path, index=False)
            logging.info(f"Raw CSV saved to {raw_csv_path} ({len(combined_df)} rows)")
        except Exception as e:
            logging.warning(f"Could not write raw CSV: {e}")

        if processing_errors:
            logging.warning(f"Completed with {len(processing_errors)} processing errors")

        return output_csv_path

    except Exception as e:
        logging.error(f"Error combining RH results: {e}")
        return None


def backfill_from_raw_csv(raw_csv_path, annual_results_dir, antenna_height_m=None):
    """Re-aggregate from an existing combined_raw.csv to produce enriched parquet.

    Use this to backfill enriched data for stations already processed --
    no need to re-run gnssir, just re-aggregate the raw per-arc data.

    Args:
        raw_csv_path: Path to existing {STATION}_{year}_combined_raw.csv
        annual_results_dir: Output directory for enriched parquet files
        antenna_height_m: Antenna height for WSE computation (optional)

    Returns:
        tuple: (per_arc_path, enriched_path) or (None, None) on failure
    """
    raw_csv_path = Path(raw_csv_path)
    annual_results_dir = Path(annual_results_dir)

    if not raw_csv_path.exists():
        logging.error(f"Raw CSV not found: {raw_csv_path}")
        return None, None

    # Parse station and year from filename: {STATION}_{year}_combined_raw.csv
    stem = raw_csv_path.stem  # e.g. "UMNQ_2025_combined_raw"
    parts = stem.split("_")
    station = parts[0]
    year = parts[1]

    logging.info(f"Backfilling enriched data for {station} {year} from {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    if df.empty:
        logging.error("Raw CSV is empty")
        return None, None

    # Ensure date column exists
    if "date" not in df.columns and "year" in df.columns and "doy" in df.columns:
        df["date"] = df.apply(
            lambda row: (
                datetime(int(row["year"]), 1, 1) + timedelta(days=int(row["doy"]) - 1)
            ).strftime("%Y-%m-%d"),
            axis=1,
        )

    add_derived_columns(df, antenna_height_m=antenna_height_m)

    annual_results_dir.mkdir(parents=True, exist_ok=True)

    # Per-arc parquet
    per_arc_path = annual_results_dir / f"{station}_{year}_per_arc.parquet"
    df.to_parquet(per_arc_path, index=False, engine="pyarrow")
    logging.info(
        f"Per-arc: {len(df)} arcs -> {per_arc_path} "
        f"({per_arc_path.stat().st_size / 1024:.0f} KB)"
    )

    # Enriched daily parquet
    enriched = compute_enriched_daily(df)
    enriched_path = None
    if enriched is not None:
        # Primary output
        enriched_path = annual_results_dir / f"{station}_{year}_combined_enriched.parquet"
        enriched.to_parquet(enriched_path, index=False, engine="pyarrow")
        # Backward-compatible alias for dashboard
        compat_path = annual_results_dir / f"{station}_{year}_daily_enriched.parquet"
        enriched.to_parquet(compat_path, index=False, engine="pyarrow")
        logging.info(
            f"Enriched: {len(enriched)} rows across "
            f"{enriched['date'].nunique()} days -> {enriched_path}"
        )

    # Interfrequency divergence parquet
    interfreq = compute_interfreq_daily(df)
    if interfreq is not None:
        interfreq_path = annual_results_dir / f"{station}_{year}_combined_interfreq.parquet"
        interfreq.to_parquet(interfreq_path, index=False, engine="pyarrow")
        logging.info(
            f"Interfreq: {len(interfreq)} rows -> {interfreq_path}"
        )

    return per_arc_path, enriched_path
