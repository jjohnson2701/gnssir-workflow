#!/usr/bin/env python3
"""
# In progress — not yet integrated
Generate self-contained HTML reports with embedded images.

Scans a directory for PNG files and builds an HTML page with base64-encoded
images, section headers, captions, and summary statistics.

Usage:
    python scripts/generate_html_report.py --title "CYGNSS FORA 2024" \
        --image_dir results_annual/FORA/cygnss \
        --output results_annual/FORA/cygnss/FORA_2024_cygnss_report.html

    python scripts/generate_html_report.py --config report_config.json
"""

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path


def img_to_base64(path):
    """Read image file and return base64 data URI."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = Path(path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "svg": "image/svg+xml"}.get(suffix.lstrip("."), "image/png")
    return f"data:{mime};base64,{data}"


def build_report(title, sections, output_path, summary_text=None):
    """Build HTML report.

    Args:
        title: Report title
        sections: list of dicts with keys:
            - heading: section heading
            - images: list of dicts with keys:
                - path: path to image file
                - caption: optional caption
                - width: optional CSS width (default "100%")
            - text: optional section text (markdown-ish, newlines → <br>)
        output_path: where to write the HTML
        summary_text: optional text for the top summary box
    """
    html_parts = []

    # Header
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    line-height: 1.6;
    padding: 20px;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{
    font-size: 1.8rem;
    margin-bottom: 8px;
    color: #1a1a2e;
    border-bottom: 3px solid #4a90d9;
    padding-bottom: 8px;
  }}
  .meta {{
    color: #666;
    font-size: 0.85rem;
    margin-bottom: 20px;
  }}
  .summary {{
    background: #e8f0fe;
    border-left: 4px solid #4a90d9;
    padding: 16px 20px;
    margin-bottom: 24px;
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
    white-space: pre-line;
  }}
  .section {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    margin-bottom: 24px;
    padding: 20px;
  }}
  .section h2 {{
    font-size: 1.3rem;
    color: #2c3e50;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #eee;
  }}
  .section-text {{
    margin-bottom: 16px;
    font-size: 0.92rem;
    color: #555;
  }}
  .figure {{
    margin-bottom: 16px;
    text-align: center;
  }}
  .figure img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #ddd;
  }}
  .figure .caption {{
    font-size: 0.85rem;
    color: #777;
    margin-top: 6px;
    font-style: italic;
  }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  @media (max-width: 800px) {{
    .two-col {{ grid-template-columns: 1fr; }}
  }}
  .footer {{
    text-align: center;
    color: #999;
    font-size: 0.8rem;
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #ddd;
  }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<div class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M UTC")} | GNSS-IR Workflow</div>
""")

    if summary_text:
        html_parts.append(f'<div class="summary">{summary_text}</div>\n')

    # Sections
    for sec in sections:
        html_parts.append('<div class="section">\n')
        if sec.get("heading"):
            html_parts.append(f'<h2>{sec["heading"]}</h2>\n')
        if sec.get("text"):
            text_html = sec["text"].replace("\n", "<br>")
            html_parts.append(f'<div class="section-text">{text_html}</div>\n')

        images = sec.get("images", [])
        use_grid = len(images) == 2 and all(
            img.get("width", "100%") == "100%" for img in images
        )

        if use_grid:
            html_parts.append('<div class="two-col">\n')

        for img in images:
            img_path = Path(img["path"])
            if not img_path.exists():
                print(f"  WARNING: {img_path} not found, skipping")
                continue
            data_uri = img_to_base64(img_path)
            width = img.get("width", "100%")
            caption = img.get("caption", "")
            html_parts.append(f'<div class="figure">\n')
            html_parts.append(f'  <img src="{data_uri}" style="width:{width};" />\n')
            if caption:
                html_parts.append(f'  <div class="caption">{caption}</div>\n')
            html_parts.append(f'</div>\n')

        if use_grid:
            html_parts.append('</div>\n')

        html_parts.append('</div>\n')

    # Footer
    html_parts.append("""
<div class="footer">
  GNSS-IR Spaceborne Comparison Report
</div>
</div>
</body>
</html>
""")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(html_parts))
    size_kb = output_path.stat().st_size / 1024
    print(f"Report saved: {output_path} ({size_kb:.0f} KB)")


def build_cygnss_report(station, year, image_dir, output_path):
    """Build a CYGNSS comparison report with standard sections."""
    d = Path(image_dir)

    summary = (
        f"Station: {station} | Year: {year}\n"
        f"Data source: CYGNSS L2 V3.2 (NASA PO.DAAC)\n"
        f"CYGNSS measures GPS signals reflected off Earth's surface from orbit — "
        f"the same physics as ground-based GNSS-IR.\n"
        f"This report compares spaceborne surface reflectivity with ground-based SNR features."
    )

    sections = [
        {
            "heading": "1. Spatial Coverage",
            "text": "Left: Fresnel coefficient gridded to 0.05° cells. The sharp boundary between high (red, ocean) "
                    "and low (blue) values traces the coastline — CYGNSS resolves land/water at ~5 km. "
                    "Right: sampling density (log scale) showing uniform coverage over open water.",
            "images": [
                {"path": str(d / f"{station}_{year}_cygnss_fresnel_heatmap.png"),
                 "caption": "Fresnel coefficient heatmap — land/water boundary clearly resolved"},
                {"path": str(d / f"{station}_{year}_cygnss_density.png"),
                 "caption": "Specular point density (log scale)"},
            ],
        },
        {
            "heading": "2. Fresnel Curve Validation",
            "text": "The empirical CYGNSS reflectivity vs incidence angle closely follows the theoretical "
                    "Fresnel curve for water (dashed navy, permittivity=80). This validates that both "
                    "CYGNSS and ground-based GNSS-IR are measuring the same physical quantity. "
                    "An ice surface (permittivity=3.2) would produce a distinctly different curve.",
            "images": [
                {"path": str(d / f"{station}_{year}_cygnss_fresnel_curve.png"),
                 "caption": "Empirical vs theoretical Fresnel curves — CYGNSS data matches water permittivity"},
            ],
        },
        {
            "heading": "3. Full-Year Time Series",
            "text": "Top: daily median Fresnel coefficient (remarkably stable at ~0.675 over open ocean). "
                    "Second: CYGNSS-derived wind speed with storm events highlighted. "
                    "Third: NBRCS (normalized bistatic radar cross section, log scale) — a roughness proxy. "
                    "Bottom: daily specular point count within 1° of station. "
                    "The red shaded region marks the window with ground-truth GNSS-IR data.",
            "images": [
                {"path": str(d / f"{station}_{year}_cygnss_seasonal.png"),
                 "caption": "CYGNSS full-year time series with ground-truth window highlighted"},
            ],
        },
        {
            "heading": "4. CYGNSS vs Ground GNSS-IR Correlations",
            "text": "Scatter plots of CYGNSS daily medians vs ground-based SNR features, colored by DOY. "
                    "Key findings: Fresnel coeff vs phase shows the strongest correlation (r=-0.29); "
                    "wind speed vs reflector height (r=-0.25) is physically expected (rougher sea = different RH). "
                    "The weak overall correlations are expected — CYGNSS averages over ~25 km while "
                    "ground GNSS-IR samples ~100 m, and the ocean surface is spectrally uniform.",
            "images": [
                {"path": str(d / f"{station}_{year}_cygnss_correlations.png"),
                 "caption": "Six key correlations between CYGNSS and ground-based features"},
            ],
        },
        {
            "heading": "5. Wind Impact on Ground Features",
            "text": "Box plots comparing ground GNSS-IR features on calm (<P20) vs windy (>P80) days "
                    "as classified by CYGNSS wind speed. RH shows the clearest response (p=0.055) — "
                    "reflector height decreases in high-wind conditions, consistent with rough-surface "
                    "scattering shifting the effective reflection point.",
            "images": [
                {"path": str(d / f"{station}_{year}_cygnss_wind_response.png"),
                 "caption": "Ground feature response to CYGNSS-derived wind conditions"},
            ],
        },
    ]

    build_report(
        title=f"CYGNSS Comparison: {station} {year}",
        sections=sections,
        output_path=output_path,
        summary_text=summary,
    )


def build_smap_report(station, year, image_dir, output_path):
    """Build a SMAP freeze/thaw comparison report."""
    d = Path(image_dir)

    summary = (
        f"Station: {station} | Year: {year}\n"
        f"Data source: SMAP Enhanced L3 Freeze/Thaw (SPL3FTP_E, 9 km daily)\n"
        f"Spatial resolution: Single 9 km EASE-Grid 2.0 pixel — the nearest cell to the station (no averaging).\n"
        f"The pixel is a mixed land/water cell (43% water) on a Lake Superior inlet, not open water.\n"
        f"SMAP L-band radiometer measures surface brightness temperature (TBv, TBh) and polarization. "
        f"Frozen surfaces have lower emissivity and different polarization signatures than "
        f"liquid water. The strong correlations (r=-0.65) between SMAP polarization and ground GNSS-IR "
        f"features confirm both instruments respond to the same freeze/thaw transitions."
    )

    sections = [
        {
            "heading": "1. Spatial Context: Where Is the SMAP Pixel?",
            "text": "Left: SMAP open water fraction on the 9 km EASE-Grid. The selected pixel "
                    "(red outline) sits on the Lake Superior north shore at 43% water fraction — "
                    "a mixed land/lake pixel that includes both the shoreline and surrounding forest. "
                    "The GNSS-IR station (red star) is ~1 km from the pixel center. "
                    "ROSS is located on an inlet where lake ice is pushed onshore during winter — "
                    "the SMAP pixel captures this mixed land/water/ice environment rather than open lake. "
                    "Pixels to the south are >60% water and receive SMAP fill values for the freeze/thaw flag. "
                    "Right: TBv brightness temperature on Jan 15 — water-dominated pixels show lower TB "
                    "(~150K, blue) vs land-dominated (~200K, red), confirming SMAP resolves the lake boundary.",
            "images": [
                {"path": str(d / f"{station}_{year}_smap_pixel_map.png"),
                 "caption": "SMAP 9 km grid cells around ROSS, colored by water fraction (left) and TBv (right)"},
            ],
        },
        {
            "heading": "1b. Multi-Scale View: SMAP Pixel to Fresnel Zone",
            "text": "Three zoom levels showing the spatial relationship. "
                    "A) Regional: SMAP 9 km grid overlaid on satellite imagery of Lake Superior north shore. "
                    "B) Station-scale: The red-shaded SMAP pixel covers the station, the inlet, and "
                    "surrounding forested shoreline. Note that the pixel is NOT open water — it's a "
                    "mixed land/water cell on an inlet where ice accumulates in winter. "
                    "C) Fresnel-zone scale: The ground GNSS-IR Fresnel zones (20-80m from antenna at different "
                    "elevations) are entirely contained within the SMAP pixel. "
                    "The correlation between SMAP and ground features therefore reflects both instruments "
                    "responding to the same freeze/thaw transitions in a mixed coastal environment — "
                    "not identical spatial sampling, but the same phenomena at the same time.",
            "images": [
                {"path": str(d / f"{station}_{year}_smap_multiscale_map.png"),
                 "caption": "A) SMAP grid on imagery  B) Station within SMAP pixel  C) Fresnel zones within pixel"},
            ],
        },
        {
            "heading": "2. Time Series: SMAP Radiometry vs GNSS-IR Classification",
            "text": "Top two panels: L-band brightness temperatures (TBv, TBh) and their polarization "
                    "difference (TBv - TBh). Frozen surfaces typically show lower absolute TB but larger "
                    "polarization difference due to increased scattering anisotropy. "
                    "Third: Normalized Polarization Ratio (NPR). "
                    "Bottom: GNSS-IR daily ice classification with ice_score overlay.",
            "images": [
                {"path": str(d / f"{station}_{year}_smap_timeseries.png"),
                 "caption": "SMAP brightness temperature vs GNSS-IR daily classification"},
            ],
        },
        {
            "heading": "3. SMAP vs Ground GNSS-IR Features",
            "text": "Direct comparison of SMAP radiometric observables with ground-based SNR features "
                    "(daily medians). Top row: SMAP polarization difference (TBv-TBh) vs ground features. "
                    "Bottom row: SMAP TBv vs ground features. Strong correlations here mean the "
                    "spaceborne 9 km L-band signal responds to the same surface changes detected "
                    "by the local ~100 m ground-based GNSS-IR Fresnel zone.",
            "images": [
                {"path": str(d / f"{station}_{year}_smap_vs_ground.png"),
                 "caption": "SMAP radiometry vs ground GNSS-IR features (colored by ice classification)"},
            ],
        },
        {
            "heading": "4. SMAP Observables by Classification",
            "text": "Box plots showing SMAP brightness temperature distributions for each "
                    "GNSS-IR classification category. If the classifier is working correctly, "
                    "ice-classified days should show systematically different TB than water-classified days.",
            "images": [
                {"path": str(d / f"{station}_{year}_smap_boxplot.png"),
                 "caption": "SMAP TB distributions grouped by GNSS-IR ice/water/transition classification"},
            ],
        },
        {
            "heading": "5. Temporal Evolution: SMAP + GNSS-IR Synchronized",
            "text": "Top 3 panels: continuous time series of SMAP polarization difference, "
                    "ground SNR variance, and ground damping (gamma), with ice/water/transition "
                    "classification as colored background shading. Vertical dotted lines mark "
                    "the snapshot dates shown in the bottom row. "
                    "Bottom row: spatial snapshots of SMAP TBv-TBh around ROSS at ~monthly intervals. "
                    "The selected pixel (black outline) and station (star) are marked. "
                    "Watch for the polarization difference to drop during ice onset (Jan-Mar) "
                    "and recover during breakup (Apr-Jun), synchronized with the ground feature changes.",
            "images": [
                {"path": str(d / f"{station}_{year}_smap_temporal_evolution.png"),
                 "caption": "Synchronized SMAP radiometry + ground GNSS-IR features with monthly spatial snapshots"},
            ],
        },
        {
            "heading": "6. Seasonal Scatter",
            "text": "SMAP observables plotted against day of year, colored by GNSS-IR classification.",
            "images": [
                {"path": str(d / f"{station}_{year}_smap_scatter.png"),
                 "caption": "SMAP observables vs season, colored by ice classification"},
            ],
        },
    ]

    build_report(
        title=f"SMAP Comparison: {station} {year}",
        sections=sections,
        output_path=output_path,
        summary_text=summary,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate HTML image report")
    parser.add_argument("--type", choices=["cygnss", "smap", "generic"], default="generic")
    parser.add_argument("--station", help="Station name")
    parser.add_argument("--year", type=int, help="Year")
    parser.add_argument("--image_dir", help="Directory containing images")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--title", help="Report title (generic mode)")
    args = parser.parse_args()

    if args.type == "cygnss":
        image_dir = args.image_dir or f"results_annual/{args.station}/cygnss"
        output = args.output or f"{image_dir}/{args.station}_{args.year}_cygnss_report.html"
        build_cygnss_report(args.station, args.year, image_dir, output)
    elif args.type == "smap":
        image_dir = args.image_dir or f"results_annual/{args.station}/smap"
        output = args.output or f"{image_dir}/{args.station}_{args.year}_smap_report.html"
        build_smap_report(args.station, args.year, image_dir, output)
    else:
        if not args.image_dir or not args.output or not args.title:
            parser.error("Generic mode requires --image_dir, --output, --title")
        # Scan for PNGs
        d = Path(args.image_dir)
        images = sorted(d.glob("*.png"))
        sections = [{"heading": "Images", "images": [
            {"path": str(p), "caption": p.stem} for p in images
        ]}]
        build_report(args.title, sections, args.output)


if __name__ == "__main__":
    main()
