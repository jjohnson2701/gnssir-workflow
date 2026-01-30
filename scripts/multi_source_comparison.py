#!/usr/bin/env python3
"""
Multi-Source GNSS-IR Comparison with External Data Integration
============================================================

Enhanced comparison module that integrates GNSS-IR data with multiple external sources:
- USGS stream/tide gauge data (existing)
- NOAA CO-OPS tide predictions and observations (new)
- NDBC buoy meteorological and wave data (new)
- Environmental correlation analysis (new)

This module extends the existing usgs_comparison.py with multi-source
validation and environmental context analysis.

Usage:
    python scripts/multi_source_comparison.py --station FORA --year 2024 --doy_start 260 --doy_end 266
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Union, Tuple, Any

# Add project modules
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import existing modules
try:
    from scripts.usgs_comparison import usgs_comparison
except ImportError:
    from usgs_comparison import usgs_comparison

try:
    from scripts.usgs_comparison_analyzer import load_gnssir_data
except ImportError:
    from usgs_comparison_analyzer import load_gnssir_data

try:
    from scripts import usgs_data_handler
    from scripts import usgs_gauge_finder
except ImportError:
    import usgs_data_handler
    import usgs_gauge_finder

try:
    from scripts.utils.config_factory import ConfigFactory
except ImportError:
    from utils.config_factory import ConfigFactory

# Import our new external APIs
try:
    from scripts.external_apis.noaa_coops import NOAACOOPSClient
    from scripts.external_apis.ndbc_client import NDBCClient
except ImportError:
    from external_apis.noaa_coops import NOAACOOPSClient
    from external_apis.ndbc_client import NDBCClient

try:
    from scripts.environmental_analysis import EnvironmentalAnalyzer
except ImportError:
    from environmental_analysis import EnvironmentalAnalyzer

# Import visualization
try:
    from scripts import visualizer
    from scripts.visualizer.comparison_plots import create_comparison_plot
except ImportError:
    import visualizer
    from visualizer.comparison_plots import create_comparison_plot

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class MultiSourceComparison:
    """
    Multi-source comparison system for GNSS-IR validation and analysis.
    
    Integrates GNSS-IR measurements with:
    - USGS gauge data (existing functionality)
    - NOAA CO-OPS tide predictions and observations
    - NDBC buoy meteorological and wave data
    - Environmental correlation analysis
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the multi-source comparison system.
        
        Args:
            log_level: Logging level for analysis operations
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize API clients
        self.coops_client = NOAACOOPSClient()
        self.ndbc_client = NDBCClient()
        self.env_analyzer = EnvironmentalAnalyzer(log_level)
        
        self.logger.info("Multi-source comparison system initialized")
    
    def run_comprehensive_analysis(
        self,
        station_name: str,
        year: int,
        doy_range: Optional[Tuple[int, int]] = None,
        max_lag_days: int = 10,
        output_dir: Optional[Path] = None,
        include_external_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive multi-source analysis.
        
        Args:
            station_name: Station name in uppercase (e.g., "FORA")
            year: Year to analyze
            doy_range: Optional range of DOYs to include (start, end)
            max_lag_days: Maximum lag to consider in days
            output_dir: Directory to save output files
            include_external_sources: Whether to include NOAA/NDBC data
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        self.logger.info(f"Starting comprehensive analysis for {station_name} {year}")
        
        # Convert year to string for consistency
        year_str = str(year)
        
        # Set up output directory
        if output_dir is None:
            output_dir = PROJECT_ROOT / "results_annual" / station_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        results = {
            'metadata': {
                'station': station_name,
                'year': year,
                'doy_range': doy_range,
                'analysis_timestamp': datetime.now().isoformat(),
                'external_sources_included': include_external_sources
            },
            'data_sources': {},
            'comparisons': {},
            'environmental_analysis': {},
            'output_files': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Step 1: Load GNSS-IR data
            gnssir_data = self._load_gnssir_data(station_name, year_str, doy_range)
            if gnssir_data is None or gnssir_data.empty:
                raise ValueError("Failed to load GNSS-IR data")
            
            results['data_sources']['gnssir'] = {
                'records': len(gnssir_data),
                'date_range': [gnssir_data['date'].min(), gnssir_data['date'].max()],
                'columns': gnssir_data.columns.tolist()
            }
            
            # Step 2: Run existing USGS comparison
            self.logger.info("Running existing USGS comparison analysis")
            usgs_results = usgs_comparison(
                station_name, year, doy_range, max_lag_days, output_dir
            )
            
            if not usgs_results.get('success', False):
                self.logger.warning(f"USGS comparison failed: {usgs_results.get('error', 'Unknown error')}")
                results['errors'].append(f"USGS comparison: {usgs_results.get('error', 'Unknown error')}")
            else:
                results['comparisons']['usgs'] = usgs_results
                results['data_sources']['usgs'] = {
                    'site_code': usgs_results.get('usgs_site_code', 'Unknown'),
                    'parameter_code': usgs_results.get('parameter_code', 'Unknown')
                }
            
            # Step 3: Load external data sources (if enabled)
            external_data = {}
            if include_external_sources:
                station_config = ConfigFactory.get_station_config(station_name)
                if station_config:
                    external_data = self._load_external_data_sources(
                        station_config, year, doy_range
                    )
                    
                    # Add external data info to results
                    for source, data in external_data.items():
                        if not data.empty:
                            results['data_sources'][source] = {
                                'records': len(data),
                                'date_range': [data['datetime'].min(), data['datetime'].max()],
                                'columns': data.columns.tolist()
                            }
            
            # Step 4: Multi-source comparison analysis
            if external_data:
                multi_source_results = self._perform_multi_source_comparison(
                    gnssir_data, external_data, station_name, year_str, output_dir
                )
                results['comparisons']['multi_source'] = multi_source_results
            
            # Step 5: Environmental analysis
            if external_data:
                env_results = self._perform_environmental_analysis(
                    gnssir_data, external_data, station_name, year_str, output_dir
                )
                results['environmental_analysis'] = env_results
            
            # Step 6: Generate comprehensive report
            report_path = self._generate_comprehensive_report(
                results, station_name, year_str, output_dir
            )
            if report_path:
                results['output_files'].append(str(report_path))
            
            self.logger.info("Comprehensive analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    def _load_gnssir_data(
        self, 
        station_name: str, 
        year: str, 
        doy_range: Optional[Tuple[int, int]]
    ) -> Optional[pd.DataFrame]:
        """Load GNSS-IR data using existing functionality."""
        try:
            gnssir_data = load_gnssir_data(station_name, year, doy_range)
            
            if gnssir_data is not None and not gnssir_data.empty:
                # Ensure datetime column exists
                if 'date' in gnssir_data.columns and 'datetime' not in gnssir_data.columns:
                    gnssir_data['datetime'] = pd.to_datetime(gnssir_data['date'])
                
                self.logger.info(f"Loaded {len(gnssir_data)} GNSS-IR records")
                return gnssir_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading GNSS-IR data: {e}")
            return None
    
    def _load_external_data_sources(
        self,
        station_config: Dict,
        year: int,
        doy_range: Optional[Tuple[int, int]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load external data sources based on station configuration.
        
        Args:
            station_config: Station configuration dictionary
            year: Year to load data for
            doy_range: Optional DOY range
            
        Returns:
            Dictionary with external data sources
        """
        external_data = {}
        
        # Calculate date range
        if doy_range:
            start_date = datetime.strptime(f"{year}-{doy_range[0]}", "%Y-%j")
            end_date = datetime.strptime(f"{year}-{doy_range[1]}", "%Y-%j")
        else:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
        
        # Get station coordinates - try multiple field names
        lat = station_config.get('latitude_deg') or station_config.get('latitude') or station_config.get('lat')
        lon = station_config.get('longitude_deg') or station_config.get('longitude') or station_config.get('lon')
        
        if lat is None or lon is None:
            self.logger.warning("Station coordinates not available - cannot load external data")
            return external_data
        
        # Load NOAA CO-OPS data
        external_sources = station_config.get('external_data_sources', {})
        
        if external_sources.get('noaa_coops', {}).get('enabled', False):
            coops_data = self._load_coops_data(
                lat, lon, start_date, end_date, external_sources['noaa_coops']
            )
            if not coops_data.empty:
                external_data['coops_tide'] = coops_data
        
        # Load NDBC data
        if external_sources.get('ndbc_buoys', {}).get('enabled', False):
            ndbc_config = external_sources['ndbc_buoys']
            
            # Load meteorological data
            met_data = self._load_ndbc_meteorological_data(
                lat, lon, ndbc_config
            )
            if not met_data.empty:
                external_data['ndbc_met'] = met_data
            
            # Load wave data
            wave_data = self._load_ndbc_wave_data(
                lat, lon, ndbc_config
            )
            if not wave_data.empty:
                external_data['ndbc_wave'] = wave_data
        
        self.logger.info(f"Loaded external data sources: {list(external_data.keys())}")
        return external_data
    
    def _load_coops_data(
        self,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime,
        coops_config: Dict
    ) -> pd.DataFrame:
        """Load NOAA CO-OPS tide data."""
        try:
            # Find nearby stations
            search_radius = coops_config.get('search_radius_km', 100)
            stations = self.coops_client.find_nearby_stations(lat, lon, search_radius)
            
            if not stations:
                self.logger.warning("No CO-OPS stations found")
                return pd.DataFrame()
            
            # Use preferred stations if specified, otherwise use nearest
            preferred_stations = coops_config.get('preferred_stations', [])
            
            if preferred_stations:
                # Try preferred stations first
                station_id = None
                for pref_id in preferred_stations:
                    if any(s['id'] == pref_id for s in stations):
                        station_id = pref_id
                        break
            
            if not station_id:
                station_id = stations[0]['id']
            
            self.logger.info(f"Using CO-OPS station: {station_id}")
            
            # Get tide predictions
            data_products = coops_config.get('data_products', ['predictions'])
            combined_data = pd.DataFrame()
            
            if 'predictions' in data_products:
                predictions = self.coops_client.get_tide_predictions(
                    station_id, start_date, end_date,
                    datum=coops_config.get('datum', 'MLLW'),
                    interval=coops_config.get('interval', 'h')
                )
                if not predictions.empty:
                    combined_data = predictions
            
            if 'water_level' in data_products:
                observations = self.coops_client.get_water_level_observations(
                    station_id, start_date, end_date,
                    datum=coops_config.get('datum', 'MLLW'),
                    interval=coops_config.get('interval', 'h')
                )
                if not observations.empty:
                    if combined_data.empty:
                        combined_data = observations
                    else:
                        # Merge predictions and observations
                        combined_data = pd.concat([combined_data, observations], ignore_index=True)
                        combined_data = combined_data.sort_values('datetime').reset_index(drop=True)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error loading CO-OPS data: {e}")
            return pd.DataFrame()
    
    def _load_ndbc_meteorological_data(
        self,
        lat: float,
        lon: float,
        ndbc_config: Dict
    ) -> pd.DataFrame:
        """Load NDBC meteorological data."""
        try:
            # Find nearby buoys
            search_radius = ndbc_config.get('search_radius_km', 200)
            buoys = self.ndbc_client.find_nearby_buoys(lat, lon, search_radius)
            
            if not buoys:
                self.logger.warning("No NDBC buoys found")
                return pd.DataFrame()
            
            # Use preferred buoys if specified, otherwise use nearest
            preferred_buoys = ndbc_config.get('preferred_buoys', [])
            
            buoy_id = None
            if preferred_buoys:
                for pref_id in preferred_buoys:
                    if any(b['id'] == pref_id for b in buoys):
                        buoy_id = pref_id
                        break
            
            if not buoy_id:
                buoy_id = buoys[0]['id']
            
            self.logger.info(f"Using NDBC buoy for meteorological data: {buoy_id}")
            
            # Get meteorological data
            days_back = ndbc_config.get('max_days_back', 45)
            met_data = self.ndbc_client.get_meteorological_data(buoy_id, days_back)
            
            return met_data
            
        except Exception as e:
            self.logger.error(f"Error loading NDBC meteorological data: {e}")
            return pd.DataFrame()
    
    def _load_ndbc_wave_data(
        self,
        lat: float,
        lon: float,
        ndbc_config: Dict
    ) -> pd.DataFrame:
        """Load NDBC wave data."""
        try:
            # Find nearby buoys (reuse logic from meteorological data)
            search_radius = ndbc_config.get('search_radius_km', 200)
            buoys = self.ndbc_client.find_nearby_buoys(lat, lon, search_radius)
            
            if not buoys:
                return pd.DataFrame()
            
            # Use preferred buoys if specified, otherwise use nearest
            preferred_buoys = ndbc_config.get('preferred_buoys', [])
            
            buoy_id = None
            if preferred_buoys:
                for pref_id in preferred_buoys:
                    if any(b['id'] == pref_id for b in buoys):
                        buoy_id = pref_id
                        break
            
            if not buoy_id:
                buoy_id = buoys[0]['id']
            
            self.logger.info(f"Using NDBC buoy for wave data: {buoy_id}")
            
            # Get wave data
            days_back = ndbc_config.get('max_days_back', 45)
            wave_data = self.ndbc_client.get_wave_data(buoy_id, days_back)
            
            return wave_data
            
        except Exception as e:
            self.logger.error(f"Error loading NDBC wave data: {e}")
            return pd.DataFrame()
    
    def _perform_multi_source_comparison(
        self,
        gnssir_data: pd.DataFrame,
        external_data: Dict[str, pd.DataFrame],
        station_name: str,
        year: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Perform multi-source comparison analysis.
        
        Args:
            gnssir_data: GNSS-IR measurements
            external_data: Dictionary of external data sources
            station_name: Station name
            year: Year string
            output_dir: Output directory
            
        Returns:
            Dictionary with multi-source comparison results
        """
        self.logger.info("Performing multi-source comparison analysis")
        
        results = {
            'correlations': {},
            'statistics': {},
            'temporal_alignment': {},
            'quality_assessment': {}
        }
        
        try:
            # Compare GNSS-IR with each external source
            for source_name, source_data in external_data.items():
                if source_data.empty:
                    continue
                
                self.logger.info(f"Analyzing correlation with {source_name}")
                
                # Temporal alignment and correlation
                correlation_result = self._calculate_multi_source_correlation(
                    gnssir_data, source_data, source_name
                )
                
                if correlation_result:
                    results['correlations'][source_name] = correlation_result
            
            # Generate multi-source comparison plot
            if results['correlations']:
                plot_path = self._create_multi_source_plot(
                    gnssir_data, external_data, station_name, year, output_dir
                )
                if plot_path:
                    results['plot_path'] = str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error in multi-source comparison: {e}")
            results['error'] = str(e)
        
        return results
    
    def _perform_environmental_analysis(
        self,
        gnssir_data: pd.DataFrame,
        external_data: Dict[str, pd.DataFrame],
        station_name: str,
        year: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Perform environmental analysis using the EnvironmentalAnalyzer.
        
        Args:
            gnssir_data: GNSS-IR measurements
            external_data: Dictionary of external data sources  
            station_name: Station name
            year: Year string
            output_dir: Output directory
            
        Returns:
            Dictionary with environmental analysis results
        """
        self.logger.info("Performing environmental analysis")
        
        try:
            # Prepare environmental data for analyzer
            env_data_for_analysis = {}
            
            # Map our external data to environmental analyzer format
            if 'ndbc_met' in external_data:
                env_data_for_analysis['buoy'] = external_data['ndbc_met']
            
            if 'ndbc_wave' in external_data:
                env_data_for_analysis['wave'] = external_data['ndbc_wave']
            
            if 'coops_tide' in external_data:
                env_data_for_analysis['tide'] = external_data['coops_tide']
            
            # Run environmental analysis
            env_results = self.env_analyzer.analyze_environmental_effects(
                gnssir_data, env_data_for_analysis, "comprehensive"
            )
            
            # Save environmental analysis report
            report_path = output_dir / f"{station_name}_{year}_environmental_analysis.json"
            
            import json
            with open(report_path, 'w') as f:
                # Convert numpy types for JSON serialization
                env_results_serializable = self._make_json_serializable(env_results)
                json.dump(env_results_serializable, f, indent=2)
            
            env_results['report_path'] = str(report_path)
            
            return env_results
            
        except Exception as e:
            self.logger.error(f"Error in environmental analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_multi_source_correlation(
        self,
        gnssir_data: pd.DataFrame,
        source_data: pd.DataFrame,
        source_name: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate correlation between GNSS-IR and external data source."""
        try:
            # Determine the value column for the external source
            value_columns = {
                'coops_tide': ['prediction_m', 'water_level_m'],
                'ndbc_met': ['wind_speed_ms', 'wave_height_m', 'pressure_hpa'],
                'ndbc_wave': ['significant_wave_height_m', 'swell_height_m']
            }
            
            possible_cols = value_columns.get(source_name, [])
            if not possible_cols:
                return None
            
            # Find the first available column
            value_col = None
            for col in possible_cols:
                if col in source_data.columns:
                    value_col = col
                    break
            
            if value_col is None:
                return None
            
            # Merge data on nearest timestamps (simple approach for now)
            merged_data = self._merge_on_timestamps(gnssir_data, source_data, value_col)
            
            if merged_data.empty:
                return None
            
            # Calculate correlation if we have RH median
            if 'rh_median_m' in merged_data.columns and value_col in merged_data.columns:
                valid_mask = ~(np.isnan(merged_data['rh_median_m']) | np.isnan(merged_data[value_col]))
                valid_data = merged_data[valid_mask]
                
                if len(valid_data) > 3:
                    from scipy.stats import pearsonr
                    correlation, p_value = pearsonr(valid_data['rh_median_m'], valid_data[value_col])
                    
                    return {
                        'correlation': correlation,
                        'p_value': p_value,
                        'n_samples': len(valid_data),
                        'value_column': value_col,
                        'significant': p_value < 0.05
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation with {source_name}: {e}")
            return None
    
    def _merge_on_timestamps(
        self,
        gnssir_data: pd.DataFrame,
        external_data: pd.DataFrame,
        value_col: str
    ) -> pd.DataFrame:
        """Simple timestamp-based merge for correlation analysis."""
        # For now, use a simple daily merge approach
        # This could be enhanced with the interpolation methods from our API clients
        
        try:
            # Ensure both have date columns
            if 'date' not in gnssir_data.columns:
                gnssir_data['date'] = pd.to_datetime(gnssir_data['datetime']).dt.date
            
            if 'date' not in external_data.columns:
                external_data['date'] = pd.to_datetime(external_data['datetime']).dt.date
            
            # Group external data by date and take mean
            daily_external = external_data.groupby('date')[value_col].mean().reset_index()
            
            # Merge on date
            merged = pd.merge(gnssir_data, daily_external, on='date', how='inner')
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging timestamps: {e}")
            return pd.DataFrame()
    
    def _create_multi_source_plot(
        self,
        gnssir_data: pd.DataFrame,
        external_data: Dict[str, pd.DataFrame],
        station_name: str,
        year: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Create a multi-source comparison plot."""
        # For now, return None - this would be implemented as a comprehensive
        # visualization showing all data sources together
        # This could use our existing visualizer modules as a base
        return None
    
    def _generate_comprehensive_report(
        self,
        results: Dict[str, Any],
        station_name: str,
        year: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Generate a comprehensive markdown report."""
        try:
            report_path = output_dir / f"{station_name}_{year}_comprehensive_report.md"
            
            with open(report_path, 'w') as f:
                f.write(f"# Comprehensive GNSS-IR Analysis Report\n\n")
                f.write(f"**Station:** {station_name}  \n")
                f.write(f"**Year:** {year}  \n")
                f.write(f"**Analysis Date:** {results['metadata']['analysis_timestamp']}  \n\n")
                
                # Data sources summary
                f.write("## Data Sources\n\n")
                for source, info in results['data_sources'].items():
                    f.write(f"### {source.upper()}\n")
                    f.write(f"- Records: {info.get('records', 'N/A')}\n")
                    f.write(f"- Date Range: {info.get('date_range', 'N/A')}\n\n")
                
                # Comparison results
                if 'comparisons' in results:
                    f.write("## Comparison Results\n\n")
                    for comp_type, comp_results in results['comparisons'].items():
                        f.write(f"### {comp_type.replace('_', ' ').title()}\n")
                        if isinstance(comp_results, dict) and comp_results.get('success'):
                            f.write("✅ Analysis completed successfully\n\n")
                        else:
                            f.write("❌ Analysis failed or incomplete\n\n")
                
                # Environmental analysis
                if 'environmental_analysis' in results and results['environmental_analysis']:
                    env_results = results['environmental_analysis']
                    if 'recommendations' in env_results:
                        f.write("## Environmental Analysis Recommendations\n\n")
                        for i, rec in enumerate(env_results['recommendations'], 1):
                            f.write(f"{i}. {rec}\n\n")
                
                # Errors and warnings
                if results['errors']:
                    f.write("## Issues and Warnings\n\n")
                    for error in results['errors']:
                        f.write(f"- ⚠️ {error}\n\n")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return None
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Multi-source GNSS-IR comparison analysis')
    parser.add_argument('--station', required=True, help='Station name (e.g., FORA)')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze')
    parser.add_argument('--doy_start', type=int, help='Start DOY (optional)')
    parser.add_argument('--doy_end', type=int, help='End DOY (optional)')
    parser.add_argument('--max_lag_days', type=int, default=10, help='Maximum lag days for analysis')
    parser.add_argument('--output_dir', help='Output directory (optional)')
    parser.add_argument('--log_level', default='INFO', help='Logging level')
    parser.add_argument('--no_external', action='store_true', help='Skip external data sources')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up DOY range
    doy_range = None
    if args.doy_start is not None and args.doy_end is not None:
        doy_range = (args.doy_start, args.doy_end)
    
    # Set up output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
    # Run analysis
    comparison = MultiSourceComparison(args.log_level)
    
    results = comparison.run_comprehensive_analysis(
        station_name=args.station.upper(),
        year=args.year,
        doy_range=doy_range,
        max_lag_days=args.max_lag_days,
        output_dir=output_dir,
        include_external_sources=not args.no_external
    )
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-SOURCE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Station: {args.station.upper()}")
    print(f"Year: {args.year}")
    print(f"Success: {'✅' if results['success'] else '❌'}")
    print(f"Data Sources: {len(results['data_sources'])}")
    print(f"Output Files: {len(results['output_files'])}")
    
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("="*60)
    
    return 0 if results['success'] else 1

if __name__ == "__main__":
    exit(main())