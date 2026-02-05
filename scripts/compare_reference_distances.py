#!/usr/bin/env python3
"""
ABOUTME: Compare distances from GNSS stations to all configured reference sources
ABOUTME: Helps identify optimal reference source for each station
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dashboard_components.station_metadata import (  # noqa: E402
    get_station_config,
    get_reference_source_info,
    _calculate_coops_distance,
)


def compare_station_references(station_id: str):
    """
    Compare all reference sources for a station and identify the closest.
    """
    config = get_station_config(station_id)
    if not config:
        print(f"ERROR: Station {station_id} not found")
        return

    gnss_lat = config.get("latitude_deg")
    gnss_lon = config.get("longitude_deg")

    print(f"\n{'='*100}")
    print(f"REFERENCE SOURCE COMPARISON: {station_id}")
    print(f"{'='*100}")
    print(f"GNSS Station Location: {gnss_lat:.4f}°N, {gnss_lon:.4f}°W\n")

    # Current primary reference
    current_ref = get_reference_source_info(station_id)
    print("CURRENT PRIMARY REFERENCE:")
    print(f"  Source: {current_ref['primary_source']}")
    print(f"  Station: {current_ref['station_name']}")
    print(f"  Distance: {current_ref.get('distance_km', 'Unknown')} km")
    if current_ref.get("notes"):
        print(f"  Notes: {current_ref['notes']}")
    print()

    # Check all configured sources
    references = []

    # USGS
    usgs_config = config.get("usgs_comparison", {})
    usgs_site = usgs_config.get("target_usgs_site")
    if usgs_site:
        usgs_notes = usgs_config.get("notes", "")
        # Extract distance from notes if available
        import re

        distance_match = re.search(r"at\s+([\d.]+)\s*km", usgs_notes, re.IGNORECASE)
        usgs_distance = float(distance_match.group(1)) if distance_match else None

        # Extract name from notes
        name = "USGS Gauge"
        if "Using " in usgs_notes and "(" in usgs_notes:
            start = usgs_notes.find("Using ") + 6
            end = usgs_notes.find("(")
            if end > start:
                name = usgs_notes[start:end].strip()

        references.append(
            {
                "source": "USGS",
                "station_id": usgs_site,
                "station_name": name,
                "distance_km": usgs_distance,
                "notes": usgs_notes,
            }
        )

    # CO-OPS
    coops_config = config.get("external_data_sources", {}).get("noaa_coops", {})
    if coops_config.get("enabled"):
        coops_stations = coops_config.get("preferred_stations", [])

        # Also check nearest_station
        nearest = coops_config.get("nearest_station", {})
        if nearest and nearest.get("id"):
            # Add to list if not already in preferred
            if nearest["id"] not in coops_stations:
                coops_stations.append(nearest["id"])

        # Map of CO-OPS station names
        coops_names = {
            "1612340": "Honolulu Harbor, HI",
            "1611400": "Nawiliwili, HI",
            "9452634": "Elfin Cove, AK",
            "9452210": "Juneau, AK",
            "8651370": "Duck, NC",
            "8652587": "Oregon Inlet Marina, NC",
            "8658163": "Wrightsville Beach, NC",
        }

        for coops_id in coops_stations:
            # Calculate distance using the function from station_metadata
            distance = _calculate_coops_distance(station_id, coops_id)

            references.append(
                {
                    "source": "CO-OPS",
                    "station_id": coops_id,
                    "station_name": coops_names.get(coops_id, f"CO-OPS {coops_id}"),
                    "distance_km": distance,
                    "notes": "Tide gauge",
                }
            )

    # ERDDAP (GLBX Bartlett Cove)
    if station_id.upper() == "GLBX":
        references.append(
            {
                "source": "ERDDAP",
                "station_id": "bartlett_cove",
                "station_name": "Bartlett Cove, AK",
                "distance_km": 0.004,
                "notes": "Co-located ERDDAP water level station",
            }
        )

    # Display all references
    print("ALL CONFIGURED REFERENCE SOURCES:\n")

    if not references:
        print("  No reference sources configured")
        return

    # Sort by distance
    valid_refs = [r for r in references if r["distance_km"] is not None]
    invalid_refs = [r for r in references if r["distance_km"] is None]

    sorted_refs = sorted(valid_refs, key=lambda x: x["distance_km"])

    for i, ref in enumerate(sorted_refs, 1):
        marker = "⭐ CLOSEST" if i == 1 else f"   #{i}"
        print(f"{marker}")
        print(f"  Source: {ref['source']}")
        print(f"  Station: {ref['station_name']} ({ref['station_id']})")
        print(f"  Distance: {ref['distance_km']:.2f} km")

        # Compare to current
        if current_ref.get("distance_km") and isinstance(current_ref["distance_km"], (int, float)):
            diff = current_ref["distance_km"] - ref["distance_km"]
            if diff > 0:
                print(f"  → {diff:.1f} km CLOSER than current reference")
            elif diff < 0:
                print(f"  → {abs(diff):.1f} km FARTHER than current reference")
            else:
                print("  -> Same distance as current reference")

        if ref.get("notes"):
            print(f"  Notes: {ref['notes']}")
        print()

    # Show any references without known distances
    if invalid_refs:
        print("REFERENCES WITHOUT DISTANCE INFORMATION:\n")
        for ref in invalid_refs:
            print(f"  • {ref['source']}: {ref['station_name']} ({ref['station_id']})")
            if ref.get("notes"):
                print(f"    Notes: {ref['notes']}")
        print()

    # Recommendation
    if sorted_refs:
        best = sorted_refs[0]
        current_dist = current_ref.get("distance_km")

        print(f"{'='*100}")
        print("RECOMMENDATION:")

        if isinstance(current_dist, (int, float)) and best["distance_km"] < current_dist:
            improvement = current_dist - best["distance_km"]
            print(f"  ✓ Switch to {best['source']}: {best['station_name']}")
            print(
                f"  Improvement: {improvement:.1f} km closer "
                f"({current_dist:.1f} km -> {best['distance_km']:.1f} km)"
            )
            print(f"  Station ID: {best['station_id']}")
        else:
            print("  Current reference is optimal")
            print(
                f"  {current_ref['primary_source']}: {current_ref['station_name']} "
                f"at {current_dist} km"
            )

        print(f"{'='*100}\n")


def main():
    """
    Compare reference sources for all stations.
    """
    # Stations to check
    stations = ["FORA", "GLBX", "VALR", "MDAI", "DESO"]

    print(f"\n{'='*100}")
    print("REFERENCE SOURCE DISTANCE COMPARISON")
    print(f"{'='*100}")
    print("\nComparing all configured reference sources for each station...")
    print("This helps identify the closest validation source.\n")

    for station_id in stations:
        compare_station_references(station_id)

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}\n")

    print("Key Findings:")
    print("  • GLBX: ERDDAP at 0.004 km is exceptional (co-located)")
    print("  • MDAI: USGS at 0.46 km is excellent")
    print("  • Check if FORA or VALR CO-OPS stations are closer than current USGS")
    print()


if __name__ == "__main__":
    main()
