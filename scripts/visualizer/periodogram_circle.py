# ABOUTME: Circular periodogram plot for one station day, next to an image of the site.
# ABOUTME: Uses gnssrefl's own arcs, periodograms and pass/fail results, so counts equal gnssir.
"""Circular periodogram plot.

One circle per signal. Angle = azimuth. Radius = reflector height. Colour =
reflection strength (periodogram amplitude). Each marker is one satellite arc
at the azimuth and height that gnssir reports:

    filled  = arc passed the gnssir quality checks
    open    = arc failed the quality checks
    small x = arc is outside the azimuth mask of the station file

The arcs, the periodograms and the pass/fail state all come from gnssrefl
(extract_arcs, strip_compute, and the result and failQC files of gnssir). Run
gnssir for the day first.

Usage:
    python -m scripts.visualizer.periodogram_circle --station fora --year 2024 --doy 60
    (REFL_CODE must point to the gnssrefl workspace)
"""
import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

FREQ_LABEL = {
    1: "GPS L1", 2: "GPS L2", 20: "GPS L2C", 5: "GPS L5",
    101: "GLONASS L1", 102: "GLONASS L2",
    201: "Galileo E1", 205: "Galileo E5a", 206: "Galileo E6", 207: "Galileo E5b", 208: "Galileo E5",
    302: "BeiDou B1", 306: "BeiDou B3", 307: "BeiDou B2",
}
# Diverging about the required amplitude: tan below the limit, neutral at it, blue above.
STRENGTH = LinearSegmentedColormap.from_list(
    "strength", ["#b89b72", "#e4d8c4", "#f5f5f5", "#9ecae1", "#3182bd", "#08306b"])
PASS, FAIL, MASKED = "#e6550d", "#444444", "#9a9a9a"


def quiet(func, *args, **kwargs):
    """Call a gnssrefl function without its screen output."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def in_mask(az, azlist):
    return any(azlist[i] <= az <= azlist[i + 1] for i in range(0, len(azlist) - 1, 2))


def load_day(station, year, doy, freqs=None):
    """Arcs with periodograms and gnssir state for one day. Returns (config, mask, rows)."""
    import gnssrefl.gps as g
    from gnssrefl.extract_arcs import extract_arcs_from_station
    from gnssrefl.gnssir_v2 import read_json_file, rewrite_azel

    cfg = quiet(read_json_file, station)
    if isinstance(cfg, tuple):
        cfg = cfg[0]
    mask = rewrite_azel(cfg.get("azval2")) or [0, 360]
    freqs = freqs or cfg["freqs"]
    lsp_method = cfg.get("lsp_method", "fast")
    buffer_hours = 2 if cfg.get("midnite", True) else 0
    # All azimuths are extracted so that the plot also shows what the mask leaves out.
    arcs = quiet(
        extract_arcs_from_station, station, year, doy, freq=freqs, buffer_hours=buffer_hours,
        dec=int(cfg.get("dec") or 1), e1=cfg["e1"], e2=cfg["e2"], ellist=cfg.get("ellist") or [],
        azlist=[0, 360], polyV=cfg["polyV"], pele=cfg["pele"], dbhz=cfg.get("dbhz", False),
        gzip=False, station_config=cfg, sat_list=cfg.get("onesat"), attach_results=["gnssir"],
        refraction_verbose=False,
    )
    rows = []
    for meta, data in arcs:
        # State as gnssir decided it: an arc with no gnssir record was outside the mask.
        res = meta.get("gnssir_processing_results")
        if res is None:
            state = "masked"
        elif np.isfinite(res.get("RH", np.nan)):
            state = "pass"
        else:
            state = "fail"
        px = pz = None
        try:
            out = quiet(g.strip_compute, data["ele"], data["snr"], meta["cf"], cfg["maxH"],
                        cfg["desiredP"], cfg["minH"], lsp_method)
            px, pz = np.atleast_1d(out[5]), np.atleast_1d(out[6])
            if len(pz) < 2 or len(pz) != len(px):
                px = pz = None
        except Exception:
            pass
        if pz is None and state == "masked":
            continue
        if state == "pass":
            rh = float(res["RH"])
        else:  # no height from gnssir: use the periodogram peak, or the inner edge if no periodogram
            rh = float(px[np.argmax(pz)]) if pz is not None else float(cfg["minH"])
        step = max(1, len(data["ele"]) // 40)
        rows.append({"freq": int(meta["freq"]), "sat": int(meta["sat"]), "az": float(meta["az_min_ele"]),
                     "px": px, "pz": pz, "rh": rh, "state": state,
                     "hour": float(meta.get("arc_timestamp", np.nan)), "edotf": float(meta.get("edot_factor", np.nan)),
                     "ele": np.asarray(data["ele"])[::step], "azi": np.asarray(data["azi"])[::step],
                     "wavelength": 2.0 * float(meta["cf"])})
    return cfg, mask, rows


def expected_heights(station, year, doy, rows, knot_hours=3.0):
    """Attach to each arc the reflector height expected from the water level at its time.

    The water level curve is a spline through the arcs that gnssir passed on the day before,
    the day itself and the day after, with the height-rate correction. No gauge is used.
    Returns the range (m) of that curve during the day, or None when too few arcs exist.
    """
    import os

    from scripts.station_check import _spline, load_results, scatter_about_spline

    d = load_results(os.environ["REFL_CODE"], station, year, doy - 1, doy + 1)
    if len(d) < 20:
        return None
    t = d[:, 1] + d[:, 4] / 24.0
    fit = scatter_about_spline(t, d[:, 2], d[:, 12], knot_hours, True)
    sp = _spline(t, fit["rh_corrected"], knot_hours)
    if sp is None:
        return None
    for r in rows:
        tt = doy + r["hour"] / 24.0
        r["expect"] = float(sp(tt) + r["edotf"] * sp.derivative()(tt) / 24.0) if np.isfinite(r["hour"]) else np.nan
    day = sp(np.linspace(max(doy, t.min()), min(doy + 1, t.max()), 200))
    return float(day.max() - day.min())


def draw_circle(ax, rows, cfg, mask, freq, req_amp, bin_deg=5.0, relative=False, half=2.0):
    """Mean periodogram per azimuth bin, with one marker per arc.

    relative=False: the radius is the reflector height (antenna above the surface).
    relative=True:  the radius is the height of the reflecting surface above (+) or below (-)
                    the water level at the time of the arc; a ring is drawn at zero.
    """
    mine = [r for r in rows if r["freq"] == freq]
    if relative:
        h = np.arange(-half, half + 1e-9, 0.02)
        lo_r, hi_r = -half, half
        mine = [r for r in mine if np.isfinite(r.get("expect", np.nan))]
        pos = lambda r: r["expect"] - r["rh"]                      # noqa: E731
        spec_of = lambda r: np.interp(h, (r["expect"] - r["px"])[::-1], r["pz"][::-1], left=np.nan, right=np.nan)  # noqa: E731
    else:
        h = np.arange(cfg["minH"], cfg["maxH"] + 1e-9, 0.02)
        lo_r, hi_r = cfg["minH"], cfg["maxH"]
        pos = lambda r: r["rh"]                                     # noqa: E731
        spec_of = lambda r: np.interp(h, r["px"], r["pz"])          # noqa: E731
    edges = np.arange(0, 360 + bin_deg, bin_deg)
    grid = np.full((len(h), len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        spec = [spec_of(r) for r in mine if r["pz"] is not None and edges[b] <= r["az"] % 360 < edges[b + 1]]
        if spec:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                grid[:, b] = np.nanmean(spec, axis=0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(lo_r - 0.35 * (hi_r - lo_r))
    ax.set_rlim(lo_r, hi_r)
    ax.set_facecolor("white")
    vmax = max(2.0 * req_amp, np.nanpercentile(grid, 99.5) if np.isfinite(grid).any() else 1.0)
    mesh = ax.pcolormesh(np.radians(edges), np.append(h, h[-1] + 0.02), np.ma.masked_invalid(grid),
                         cmap=STRENGTH, norm=TwoSlopeNorm(vmin=0, vcenter=req_amp, vmax=vmax),
                         shading="flat", rasterized=True)
    # Azimuths that the station file leaves out: light hatch
    for i in range(len(mask) // 2 + 1):
        lo = mask[2 * i - 1] if i > 0 else 0.0
        hi = mask[2 * i] if 2 * i < len(mask) else 360.0
        if hi - lo > 0.5:
            th = np.radians(np.linspace(lo, hi, 60))
            ax.fill_between(th, lo_r, hi_r, facecolor="none", edgecolor="#c8c8c8",
                            hatch="////", linewidth=0, zorder=2)
    if relative:
        ax.plot(np.linspace(0, 2 * np.pi, 361), np.zeros(361), color="#111111", lw=1.0, ls=(0, (4, 2)), zorder=4)
    for state, kw in (
        ("masked", dict(marker="x", ms=3.5, mec=MASKED, mew=0.8, ls="none")),
        ("fail", dict(marker="o", ms=5, mfc="none", mec=FAIL, mew=1.0, ls="none")),
        ("pass", dict(marker="o", ms=5.5, mfc=PASS, mec="white", mew=0.7, ls="none")),
    ):
        sel = [r for r in mine if r["state"] == state and lo_r <= pos(r) <= hi_r]
        if sel:
            ax.plot(np.radians([r["az"] for r in sel]), [pos(r) for r in sel], zorder=5, **kw)
    n = {s: sum(r["state"] == s for r in mine) for s in ("pass", "fail", "masked")}
    if relative:
        near = sum(r["state"] == "pass" and abs(pos(r)) <= 0.5 for r in mine)
        ax.set_title(f"{FREQ_LABEL.get(freq, freq)}\n{near} of {n['pass']} passed arcs within 0.5 m of the water level",
                     fontsize=10, pad=12)
        ax.set_yticks([t for t in (-half, -half / 2, 0, half / 2, half)])
        ax.set_yticklabels([f"{t:+g}" if t else "0" for t in (-half, -half / 2, 0, half / 2, half)])
    else:
        ax.set_title(f"{FREQ_LABEL.get(freq, freq)}\n{n['pass']} pass, {n['fail']} fail, "
                     f"{n['masked']} outside mask", fontsize=10, pad=12)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "", "", ""], fontsize=9)
    ax.set_rlabel_position(next((0.5 * (mask[2 * i - 1] + mask[2 * i]) for i in range(1, len(mask) // 2)
                                 if mask[2 * i] - mask[2 * i - 1] > 20), 135))
    ax.tick_params(axis="y", labelsize=7, colors="#555555")
    ax.grid(color="#bbbbbb", lw=0.4, alpha=0.6)
    return mesh, n


def fresnel_zone(wavelength, ele_deg, h):
    """First Fresnel zone on a flat horizontal surface (Larson and Nievinski 2013, appendix;
    same formulas as gnssrefl refl_zones.FresnelZone). Returns semi-major a, semi-minor b,
    and centre distance R, in metres. a lies along the satellite azimuth."""
    d = 0.5 * wavelength
    s = np.sin(np.radians(ele_deg))
    b = np.sqrt(2 * d * h / s + (d / s) ** 2)
    return b / s, b, (h + d / s) / np.tan(np.radians(ele_deg))


US_BOXES = ((24.0, 50.0, -125.5, -66.0), (51.0, 72.0, -180.0, -129.0), (18.5, 22.5, -161.0, -154.5),
            (17.5, 18.7, -67.5, -64.5))  # lat1, lat2, lon1, lon2: lower 48, Alaska, Hawaii, Puerto Rico


def imagery_source(lat, lon, imagery="auto"):
    """Tile provider, credit text and highest useful zoom for the site image.

    "usgs" is the USGS National Map imagery (public domain, United States only). "esri" is
    Esri World Imagery (worldwide; Esri terms of use apply). "auto" takes USGS inside the
    United States and Esri elsewhere. "none" gives no image.
    """
    import contextily as cx

    if imagery == "auto":
        imagery = "usgs" if any(a <= lat <= b and c <= lon <= d for a, b, c, d in US_BOXES) else "esri"
    if imagery == "usgs":
        return cx.providers.USGS.USImagery, "Imagery: USGS, The National Map", 16
    if imagery == "esri":
        return cx.providers.Esri.WorldImagery, "Imagery: Esri World Imagery", 18
    return None, "no imagery", None


NAIP_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"


def naip_image(x1, y1, x2, y2, pixels=800):
    """Air photo (USDA NAIP, about 1 m, public domain) for a web-mercator box, from the USGS
    image service. Returns an RGB array, or None where NAIP has no cover (it covers the
    lower 48 states)."""
    import io

    import requests
    from PIL import Image

    r = requests.get(NAIP_URL, timeout=60, params={"bbox": f"{x1},{y1},{x2},{y2}", "bboxSR": 3857, "imageSR": 3857,
                                                   "size": f"{pixels},{pixels}", "format": "png", "f": "image"})
    if not r.ok or "image" not in r.headers.get("content-type", ""):
        return None
    img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
    return img if img.std() > 2 else None   # one flat colour = no cover


def draw_site(ax, cfg, mask, rows, half_width_m=None, imagery="auto"):
    """Image of the site with the reflection distances and the azimuth mask."""
    import contextily as cx
    from pyproj import Transformer

    passed = [r["rh"] for r in rows if r["state"] == "pass"]
    rh = float(np.median(passed)) if passed else 0.5 * (cfg["minH"] + cfg["maxH"])
    # Rings: inner and outer reach of the first Fresnel zone over the elevation range,
    # for the longest wavelength shown.
    lam = max([r["wavelength"] for r in rows] or [0.2548])
    a1, _, r1 = fresnel_zone(lam, cfg["e1"], rh)
    a2, _, r2 = fresnel_zone(lam, cfg["e2"], rh)
    d_far, d_near = r1 + a1, max(r2 - a2, 0.0)
    half = half_width_m or 1.12 * d_far
    x0, y0 = Transformer.from_crs(4326, 3857, always_xy=True).transform(cfg["lon"], cfg["lat"])
    k = 1.0 / np.cos(np.radians(cfg["lat"]))  # web-mercator metres per ground metre
    ax.set_xlim(x0 - half * k, x0 + half * k)
    ax.set_ylim(y0 - half * k, y0 + half * k)
    provider, source, max_zoom = imagery_source(cfg["lat"], cfg["lon"], imagery)
    ax.set_facecolor("#55606b")
    if provider is not None:
        try:
            naip = naip_image(x0 - half * k, y0 - half * k, x0 + half * k, y0 + half * k) if "USGS" in source else None
        except Exception:
            naip = None
        if naip is not None:
            ax.imshow(naip, extent=(x0 - half * k, x0 + half * k, y0 - half * k, y0 + half * k), zorder=0)
            source, provider = "Imagery: USDA NAIP, from USGS The National Map", None
    if provider is not None:
        try:
            # tiles for about 800 pixels across the image, not above the provider's best zoom
            zoom = int(np.clip(np.floor(np.log2(40075016.7 * 800 / (256 * 2 * half * k))), 1, max_zoom))
            cx.add_basemap(ax, source=provider, attribution=False, zoom=zoom)
        except Exception as exc:  # no network: keep the geometry, say so
            source = f"no imagery ({type(exc).__name__})"
    th = np.radians(np.linspace(0, 360, 361))
    for d, ls in ((d_near, "-"), (d_far, "-")):
        ax.plot(x0 + d * k * np.sin(th), y0 + d * k * np.cos(th), color="white", lw=1.0, ls=ls, alpha=0.9)
    # Light overlay: first Fresnel zones of every arc that passed, at six elevations along the arc
    t = np.linspace(0, 2 * np.pi, 60)
    for r in rows:
        if r["state"] != "pass":
            continue
        for j in np.linspace(0, len(r["ele"]) - 1, 6).astype(int):
            sa, sb, rc = fresnel_zone(r["wavelength"], r["ele"][j], r["rh"])
            az = np.radians(r["azi"][j])
            along, across = rc + sa * np.cos(t), sb * np.sin(t)
            east = along * np.sin(az) + across * np.cos(az)
            north = along * np.cos(az) - across * np.sin(az)
            ax.fill(x0 + east * k, y0 + north * k, facecolor="#ffd27f", edgecolor="none", alpha=0.07)
    wraps = (0 in mask or 0.0 in mask) and (360 in mask or 360.0 in mask)
    for i in range(0, len(mask) - 1, 2):
        for a in (mask[i], mask[i + 1]):
            if not (wraps and a in (0, 360)):
                ax.plot([x0 + d_near * k * np.sin(np.radians(a)), x0 + d_far * k * np.sin(np.radians(a))],
                        [y0 + d_near * k * np.cos(np.radians(a)), y0 + d_far * k * np.cos(np.radians(a))],
                        color="white", lw=1.0)
    ax.plot(x0, y0, marker="v", ms=9, mfc=PASS, mec="white", mew=1.0)
    bar = 10 ** np.floor(np.log10(half))
    bar = bar * (5 if half / bar > 6 else 2 if half / bar > 2.5 else 1)
    bx, by = x0 - 0.9 * half * k, y0 - 0.9 * half * k
    ax.plot([bx, bx + bar * k], [by, by], color="white", lw=3, solid_capstyle="butt")
    ax.text(bx + 0.5 * bar * k, by + 0.03 * half * k, f"{bar:g} m", color="white", ha="center",
            va="bottom", fontsize=8)
    ax.text(x0, y0 + 0.93 * half * k, "N", color="white", ha="center", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    n_pass = sum(r["state"] == "pass" for r in rows)
    ax.set_title(f"Site (height {rh:.1f} m, {cfg['e2']:g} to {cfg['e1']:g} deg)\n"
                 f"rings: reach of first Fresnel zone, {d_near:.0f} to {d_far:.0f} m\n"
                 f"shading: first Fresnel zones of the\n{n_pass} arcs that passed (most of the sensitivity)",
                 fontsize=8.5, pad=8)
    ax.text(0.99, 0.01, source, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="white")


def make_figure(station, year, doy, out, freqs=None, with_site=True, tidal_m=0.5, imagery="auto"):
    t0 = time.time()
    cfg, mask, rows = load_day(station, year, doy, freqs)
    t_data = time.time() - t0
    freqs = freqs or [f for f in cfg["freqs"] if any(r["freq"] == f for r in rows)]
    try:
        day_range = expected_heights(station, year, doy, rows)
    except Exception:
        day_range = None
    # Which view comes first: where the surface moves by more than tidal_m in the day, the view
    # relative to the water level; elsewhere the reflector height.
    views = ["height"] if day_range is None else (["water", "height"] if day_range > tidal_m else ["height", "water"])
    head = {
        "height": ("HEIGHT ABOVE THE SURFACE",
                   f"radius = reflector height: how far the antenna is above the reflecting surface,\n"
                   f"{cfg['minH']:g} m at the inner edge to {cfg['maxH']:g} m at the outer edge"),
        "water": ("DISTANCE FROM THE WATER LEVEL",
                  "radius = height of the reflecting surface above (+) or below (-) the water level at the time of the pass;\n"
                  "dashed ring = on the water. The water level comes from this station's passed arcs, not from a gauge."),
    }
    ncol = len(freqs) + (1 if with_site else 0)
    nrow = len(views)
    fig = plt.figure(figsize=(4.6 * ncol, 6.6 * nrow + 0.8))
    top, bottom = 1 - 2.0 / (6.6 * nrow + 0.8), 0.9 / (6.6 * nrow + 0.8)
    gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.98, top=top, bottom=bottom, hspace=0.62, wspace=0.18)
    if with_site:
        draw_site(fig.add_subplot(gs[0, 0]), cfg, mask, rows, imagery=imagery)
    counts = {}
    for i, view in enumerate(views):
        first = 1 if with_site else 0
        for k, f in enumerate(freqs):
            ax = fig.add_subplot(gs[i, first + k], projection="polar")
            req = cfg["reqAmp"][cfg["freqs"].index(f)] if f in cfg["freqs"] else cfg["reqAmp"][0]
            mesh, n = draw_circle(ax, rows, cfg, mask, f, req, relative=(view == "water"))
            if view == "height":
                counts[f] = n
            cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.045, pad=0.11)
            vmax = mesh.norm.vmax
            cb.set_ticks([0, req, vmax])
            cb.set_ticklabels(["0", f"{req:g}\nrequired", f"{vmax:.0f}"])
            cb.set_label("reflection strength (periodogram amplitude, volts/volts)\ntan = below the required amplitude, blue = above", fontsize=7.5)
            cb.ax.tick_params(labelsize=7)
            cb.ax.axvline(req, color="#222222", lw=1.2)
        # row heading, above the circles of this row
        box = gs[i, first].get_position(fig)
        inch = 1.0 / (6.6 * nrow + 0.8)
        fig.text(box.x0, box.y1 + 1.22 * inch, head[view][0], fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(box.x0, box.y1 + 0.72 * inch, head[view][1], fontsize=8.5, ha="left", va="bottom", color="#333333")
    note = "" if day_range is None else f"   The water level moved by {day_range:.1f} m on this day."
    fig.suptitle(
        f"{station.upper()}  {year} day {doy:03d}    angle = azimuth; elevation {cfg['e1']:g} to {cfg['e2']:g} deg.{note}\n"
        "filled = arc passed gnssir checks;  open = failed;  x = outside the azimuth mask (hatched)",
        fontsize=10, y=0.995)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"{out}  (gnssrefl arcs and periodograms {t_data:.1f} s, total {time.time() - t0:.1f} s)")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--station", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--doy", type=int, required=True)
    ap.add_argument("--freqs", help="comma list of gnssrefl frequency codes; default: all in the station file")
    ap.add_argument("--out")
    ap.add_argument("--no_site", action="store_true", help="leave out the site image")
    ap.add_argument("--imagery", default="auto", choices=["auto", "usgs", "esri", "none"],
                    help="site image: USGS (United States, public domain), Esri (worldwide), auto (default) or none")
    a = ap.parse_args()
    freqs = [int(x) for x in a.freqs.split(",")] if a.freqs else None
    out = a.out or f"{a.station.lower()}_{a.year}_{a.doy:03d}_periodogram_circle.png"
    make_figure(a.station.lower(), a.year, a.doy, out, freqs, not a.no_site, imagery=a.imagery)


if __name__ == "__main__":
    sys.exit(main())
