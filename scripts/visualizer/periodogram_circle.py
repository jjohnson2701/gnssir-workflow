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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FREQ_LABEL = {
    1: "GPS L1", 2: "GPS L2", 20: "GPS L2C", 5: "GPS L5",
    101: "GLONASS L1", 102: "GLONASS L2",
    201: "Galileo E1", 205: "Galileo E5a", 206: "Galileo E6", 207: "Galileo E5b", 208: "Galileo E5",
    302: "BeiDou B1", 306: "BeiDou B3", 307: "BeiDou B2",
}
STRENGTH = LinearSegmentedColormap.from_list("strength", ["#f4f6f8", "#9ecae1", "#3182bd", "#08306b"])
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
        dec=int(cfg.get("dec", 1)), e1=cfg["e1"], e2=cfg["e2"], ellist=cfg.get("ellist") or [],
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
        rows.append({"freq": int(meta["freq"]), "sat": int(meta["sat"]), "az": float(meta["az_min_ele"]),
                     "px": px, "pz": pz, "rh": rh, "state": state})
    return cfg, mask, rows


def draw_circle(ax, rows, cfg, mask, freq, req_amp, bin_deg=5.0):
    """Mean periodogram per azimuth bin, with one marker per arc."""
    h = np.arange(cfg["minH"], cfg["maxH"] + 1e-9, 0.02)
    edges = np.arange(0, 360 + bin_deg, bin_deg)
    grid = np.full((len(h), len(edges) - 1), np.nan)
    mine = [r for r in rows if r["freq"] == freq]
    for b in range(len(edges) - 1):
        spec = [np.interp(h, r["px"], r["pz"]) for r in mine
                if r["pz"] is not None and edges[b] <= r["az"] % 360 < edges[b + 1]]
        if spec:
            grid[:, b] = np.mean(spec, axis=0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(cfg["minH"] - 0.35 * (cfg["maxH"] - cfg["minH"]))
    ax.set_rlim(cfg["minH"], cfg["maxH"])
    ax.set_facecolor("white")
    vmax = max(2.0 * req_amp, np.nanpercentile(grid, 99.5) if np.isfinite(grid).any() else 1.0)
    mesh = ax.pcolormesh(np.radians(edges), np.append(h, h[-1] + 0.02), np.ma.masked_invalid(grid),
                         cmap=STRENGTH, vmin=0, vmax=vmax, shading="flat", rasterized=True)
    # Azimuths that the station file leaves out: light hatch
    for i in range(len(mask) // 2 + 1):
        lo = mask[2 * i - 1] if i > 0 else 0.0
        hi = mask[2 * i] if 2 * i < len(mask) else 360.0
        if hi - lo > 0.5:
            th = np.radians(np.linspace(lo, hi, 60))
            ax.fill_between(th, cfg["minH"], cfg["maxH"], facecolor="none", edgecolor="#c8c8c8",
                            hatch="////", linewidth=0, zorder=2)
    for state, kw in (
        ("masked", dict(marker="x", ms=3.5, mec=MASKED, mew=0.8, ls="none")),
        ("fail", dict(marker="o", ms=5, mfc="none", mec=FAIL, mew=1.0, ls="none")),
        ("pass", dict(marker="o", ms=5.5, mfc=PASS, mec="white", mew=0.7, ls="none")),
    ):
        sel = [r for r in mine if r["state"] == state]
        if sel:
            ax.plot(np.radians([r["az"] for r in sel]), [r["rh"] for r in sel], zorder=5, **kw)
    n = {s: sum(r["state"] == s for r in mine) for s in ("pass", "fail", "masked")}
    ax.set_title(f"{FREQ_LABEL.get(freq, freq)}\n{n['pass']} pass, {n['fail']} fail, "
                 f"{n['masked']} outside mask", fontsize=10, pad=12)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=9)
    ax.set_rlabel_position(next((0.5 * (mask[2 * i - 1] + mask[2 * i]) for i in range(1, len(mask) // 2)
                                 if mask[2 * i] - mask[2 * i - 1] > 20), 135))
    ax.tick_params(axis="y", labelsize=7, colors="#555555")
    ax.grid(color="#bbbbbb", lw=0.4, alpha=0.6)
    return mesh, n


def draw_site(ax, cfg, mask, rows, half_width_m=None):
    """Image of the site with the reflection distances and the azimuth mask."""
    import contextily as cx
    from pyproj import Transformer

    passed = [r["rh"] for r in rows if r["state"] == "pass"]
    rh = float(np.median(passed)) if passed else 0.5 * (cfg["minH"] + cfg["maxH"])
    d_far = rh / np.tan(np.radians(cfg["e1"]))
    d_near = rh / np.tan(np.radians(cfg["e2"]))
    half = half_width_m or 1.25 * d_far
    x0, y0 = Transformer.from_crs(4326, 3857, always_xy=True).transform(cfg["lon"], cfg["lat"])
    k = 1.0 / np.cos(np.radians(cfg["lat"]))  # web-mercator metres per ground metre
    ax.set_xlim(x0 - half * k, x0 + half * k)
    ax.set_ylim(y0 - half * k, y0 + half * k)
    source = "Esri World Imagery"
    try:
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution=False)
    except Exception as exc:  # no network: keep the geometry, say so
        source = f"no imagery ({type(exc).__name__})"
        ax.set_facecolor("#dddddd")
    th = np.radians(np.linspace(0, 360, 361))
    for d, ls in ((d_near, "-"), (d_far, "-")):
        ax.plot(x0 + d * k * np.sin(th), y0 + d * k * np.cos(th), color="white", lw=1.0, ls=ls, alpha=0.9)
    for i in range(0, len(mask) - 1, 2):
        for a in (mask[i], mask[i + 1]):
            if not (mask[i] == 0 and mask[-1] == 360 and a in (0, 360)):
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
    ax.set_title(f"Site\nrings: reflections from {d_near:.0f} to {d_far:.0f} m\n"
                 f"(height {rh:.1f} m, elevation {cfg['e2']:g} to {cfg['e1']:g} deg)", fontsize=9, pad=8)
    ax.text(0.99, 0.01, source, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="white")


def make_figure(station, year, doy, out, freqs=None, with_site=True):
    t0 = time.time()
    cfg, mask, rows = load_day(station, year, doy, freqs)
    t_data = time.time() - t0
    freqs = freqs or [f for f in cfg["freqs"] if any(r["freq"] == f for r in rows)]
    ncol = len(freqs) + (1 if with_site else 0)
    fig = plt.figure(figsize=(4.6 * ncol, 5.6))
    col = 1
    if with_site:
        draw_site(fig.add_subplot(1, ncol, col), cfg, mask, rows)
        col += 1
    counts = {}
    for f in freqs:
        ax = fig.add_subplot(1, ncol, col, projection="polar")
        req = cfg["reqAmp"][cfg["freqs"].index(f)] if f in cfg["freqs"] else cfg["reqAmp"][0]
        mesh, counts[f] = draw_circle(ax, rows, cfg, mask, f, req)
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08)
        cb.set_label("reflection strength (periodogram amplitude, volts/volts)", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        cb.ax.axvline(req, color=PASS, lw=1.5)
        cb.ax.text(req, 1.25, "required", color=PASS, fontsize=7, ha="center", va="bottom",
                   transform=cb.ax.get_xaxis_transform())
        col += 1
    fig.suptitle(
        f"{station.upper()}  {year} day {doy:03d}    angle = azimuth, radius = reflector height "
        f"{cfg['minH']:g} to {cfg['maxH']:g} m, elevation {cfg['e1']:g} to {cfg['e2']:g} deg\n"
        "filled = arc passed gnssir checks;  open = failed;  x = outside the azimuth mask (hatched)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
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
    a = ap.parse_args()
    freqs = [int(x) for x in a.freqs.split(",")] if a.freqs else None
    out = a.out or f"{a.station.lower()}_{a.year}_{a.doy:03d}_periodogram_circle.png"
    make_figure(a.station.lower(), a.year, a.doy, out, freqs, not a.no_site)


if __name__ == "__main__":
    sys.exit(main())
