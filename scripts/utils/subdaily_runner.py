# ABOUTME: Runs gnssrefl subdaily analysis and saves annotated output with confidence flags.
# ABOUTME: Bridges gnssrefl's subdaily spline fitting into the station processing pipeline.

import logging
import subprocess
from pathlib import Path
from typing import Optional

from scripts.utils.subdaily_loader import (
    annotate_confidence,
    load_observation_times,
    load_subdaily_results,
)

logger = logging.getLogger(__name__)


def find_spline_output(station: str, refl_code_base: Path) -> Optional[Path]:
    """
    Locate the subdaily spline output file.

    Args:
        station: Station name (lowercase)
        refl_code_base: Path to REFL_CODE directory

    Returns:
        Path to spline output file, or None if not found
    """
    path = refl_code_base / "Files" / station / f"{station}_spline_out.txt"
    return path if path.exists() else None


def find_observation_file(station: str, year: int, refl_code_base: Path) -> Optional[Path]:
    """
    Locate the IF-corrected observation file produced by subdaily.

    Args:
        station: Station name (lowercase)
        year: Processing year
        refl_code_base: Path to REFL_CODE directory

    Returns:
        Path to observation file, or None if not found
    """
    path = (
        refl_code_base
        / "Files"
        / station
        / f"{station}_{year}_subdaily_edit.txt.withrhdotIF"
    )
    return path if path.exists() else None


def process_subdaily_output(
    station: str,
    year: int,
    refl_code_base: Path,
    output_dir: Path,
) -> Optional[Path]:
    """
    Load subdaily spline output, annotate with confidence, and save as CSV.

    Args:
        station: Station name (lowercase)
        year: Processing year
        refl_code_base: Path to REFL_CODE directory
        output_dir: Directory to write output CSV

    Returns:
        Path to output CSV, or None if spline output not found
    """
    spline_path = find_spline_output(station, refl_code_base)
    if spline_path is None:
        logger.warning(f"No subdaily spline output found for {station}")
        return None

    df = load_subdaily_results(spline_path)
    logger.info(f"Loaded {len(df)} spline points from {spline_path.name}")

    # Add confidence annotations if observation file is available
    obs_path = find_observation_file(station, year, refl_code_base)
    if obs_path is not None:
        obs_times = load_observation_times(obs_path)
        df = annotate_confidence(df, obs_times)
        logger.info(
            f"Added confidence annotations from {len(obs_times)} observations"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{station}_{year}_subdaily.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved subdaily output to {output_path}")

    return output_path


def run_subdaily_command(
    station: str,
    year: int,
    doy_start: int = 1,
    doy_end: int = 366,
    plt: bool = False,
    knots: int = 8,
) -> bool:
    """
    Run gnssrefl's subdaily command.

    Args:
        station: Station name (lowercase)
        year: Processing year
        doy_start: Start day of year
        doy_end: End day of year
        plt: Whether to show plots (False for automation)
        knots: Knots per day for spline fit (8 for tidal sites)

    Returns:
        True if command succeeded
    """
    cmd = [
        "subdaily",
        station,
        str(year),
        "-plt", str(plt),
        "-knots", str(knots),
        "-doy1", str(doy_start),
        "-doy2", str(doy_end),
    ]

    logger.info(f"Running subdaily: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("subdaily completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"subdaily failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        logger.error("subdaily command not found - is gnssrefl installed?")
        return False
