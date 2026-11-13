# ABOUTME: Builds one static HTML results page for a list of stations from gnssrefl result files.
# ABOUTME: Per station: circular plot, water level series, peak-to-noise limit plot, station checks.
"""Static station results page.

The page has a summary table and one section per station with the same four blocks:
where the reflections come from, the water level series, the effect of the
peak-to-noise limit, height quality against wind (optional), and the station checks. All figures are embedded, so the page is
one file.

Input is a small YAML file. Only `stations` and the station names are required.

    year: 2024
    doy: 60              # first day of the period for the numbers
    doy_end: 91
    circle_doy: 70       # day of the circular plot
    series_days: [70, 77]
    title: GNSS-IR Station Results
    stations:
      glbx:
        place: Gustavus, Alaska
        role: Accuracy test at very low elevation.
        receiver: Trimble Alloy, 1 s files
        gauge_csv: path/to/gauge.csv         # optional; needs a time column and a level column (m)
        gauge_time: time
        gauge_level: water_surface_above_navd88
        gauge_name: Bartlett Cove gauge, 0.3 km
        inverse_csv: path/to/inverse.csv     # optional; output of scripts.inverse_heights
        open_extension: open                 # optional; gnssir run with the limits off, for the limit plot
        rinex: path/to/file.rnx              # optional; for the L2 signal check
        wind_csv: path/to/wind.csv           # optional, with a gauge; needs a time column and a speed column (m/s)
        wind_time: time
        wind_speed: speed
        wind_name: sensor at the gauge
        notes: ["free text line", "another line"]

Usage:
    python -m scripts.station_page --spec page.yaml --out results.html [--fragment]
    (REFL_CODE must point to the gnssrefl workspace)
"""
import argparse
import base64
import datetime
import glob
import html
import json
import os
from pathlib import Path

import numpy as np

from scripts.station_check import (
    L2_VERDICT_TEXT,
    l2_signal_check,
    load_results,
    read_snr,
    sampling_check,
    scatter_about_spline,
)

CSS = """
:root{--bg:#fbfcfd;--panel:#f1f4f7;--fg:#18222c;--muted:#5a6875;--line:#d5dce3;--accent:#1b6fb5;
 --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12181e;--panel:#1a232c;--fg:#e4eaf0;--muted:#9aa8b5;--line:#2e3a46;--accent:#7db8e0;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12181e;--panel:#1a232c;--fg:#e4eaf0;--muted:#9aa8b5;--line:#2e3a46;--accent:#7db8e0;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:16px;line-height:1.55;margin:0;
 padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 64px}
main{max-width:980px;margin:0 auto;display:flex;flex-direction:column;gap:20px;min-width:0}
section{display:flex;flex-direction:column;gap:14px;min-width:0;border-top:1px solid var(--line);padding-top:22px;margin-top:10px}
h1{font-size:1.9rem;line-height:1.2;margin:0;font-weight:600;text-wrap:balance}
h2{font-size:1.5rem;margin:0;font-weight:600;font-family:var(--mono);letter-spacing:.02em}
h2 .place{font-family:var(--sans);font-weight:400;font-size:1rem;color:var(--muted);letter-spacing:0;margin-left:.5em}
h3{font-size:1.02rem;margin:8px 0 0;font-weight:600}
p{margin:0;max-width:72ch}.lede{font-size:1.05rem}.meta,.sub{color:var(--muted);font-size:.86rem}.sub{font-weight:400}.role{color:var(--muted)}
a{color:var(--accent)}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;min-width:0}
table{border-collapse:collapse;width:100%;font-size:.92rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{background:var(--panel);font-weight:600}tbody th{font-family:var(--mono);font-weight:500}tbody th .sub{font-family:var(--sans)}
tr:last-child td,tr:last-child th{border-bottom:0}td.n{text-align:right;white-space:nowrap}
figure{margin:0;display:flex;flex-direction:column;gap:8px;min-width:0}
.figbox{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:#ffffff}
.figbox img{display:block;width:100%;max-width:100%}.figbox.wide img{min-width:900px;max-width:none}
figcaption{color:var(--muted);font-size:.88rem;max-width:80ch}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px 18px;margin:0;padding:12px 14px;background:var(--panel);border-radius:6px}
.facts div{min-width:0}dt{font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}dd{margin:0;font-variant-numeric:tabular-nums}
.pair{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:16px;align-items:start}
@media (max-width:720px){.pair{grid-template-columns:minmax(0,1fr)}}
ul{margin:0;padding-left:1.2em;display:flex;flex-direction:column;gap:6px;max-width:76ch}code{font-family:var(--mono);font-size:.88em}
"""

CIRCLE_CAPTION = (
    "Day {doy:03d}. Left: site image with the first Fresnel zones of the arcs that passed. Right: two rows of circles, "
    "one circle per signal. In every circle the angle is azimuth and the colour is reflection strength about the "
    "amplitude limit. <b>Height above the surface:</b> the radius is the reflector height, the distance of the antenna "
    "above the reflecting surface. <b>Distance from the water level:</b> the radius is the height of the reflecting "
    "surface above or below the water level at the time of the pass, and the dashed ring is the water. The water level "
    "comes from this station's own passed arcs. The water-level row is first where the water moved by more than 0.5 m "
    "in the day. Filled marker: the arc passed. Open marker: it failed. Hatched: outside the azimuth mask."
)


def read_gauge(path, time_col, level_col, year):
    """Gauge file -> (day of year with fraction, level in m). Rows that do not parse are dropped."""
    import pandas as pd

    g = pd.read_csv(path)
    t = pd.to_datetime(g[time_col], utc=True, errors="coerce").dt.tz_localize(None)
    v = pd.to_numeric(g[level_col], errors="coerce")
    ok = t.notna() & v.notna()
    t, v = t[ok], v[ok]
    order = np.argsort(t.values)
    doy = ((t - pd.Timestamp(f"{year}-01-01")).dt.total_seconds() / 86400 + 1).values[order]
    return doy, v.values[order]


def gauge_fit(t, rh, edotf, gauge):
    """Slope and robust scatter (m) of single heights against the gauge, with the height-rate term."""
    from scripts.visualizer.station_figures import inside

    ok = inside(t, gauge[0]) & np.isfinite(rh) & np.isfinite(edotf)
    t, rh, edotf = t[ok], rh[ok], edotf[ok]
    wl = np.interp(t, *gauge)
    rate = (np.interp(t + 300 / 86400, *gauge) - np.interp(t - 300 / 86400, *gauge)) / 600 * 3600
    A = np.c_[wl, edotf * rate, np.ones(len(t))]
    c = np.linalg.lstsq(A, -rh, rcond=None)[0]
    r = -rh - A @ c
    return float(c[0]), float(1.4826 * np.median(np.abs(r - np.median(r))))


def station_block(sta, spec, top, refl_code, fig_dir):
    """Compute the numbers and make the figures of one station. Returns a dict for the page."""
    import pandas as pd

    from scripts.visualizer.periodogram_circle import make_figure
    from scripts.visualizer.station_figures import (
        series_figure,
        tuning_figure,
        tuning_table,
        wind_figure,
        wind_table,
    )

    year, d1, d2 = top["year"], top["doy"], top["doy_end"]
    cfg = json.loads(Path(refl_code, "input", f"{sta}.json").read_text())
    d = load_results(refl_code, sta, year, d1, d2)
    if len(d) == 0:
        raise SystemExit(
            f"No gnssir results for {sta} {year} days {d1:03d} to {d2:03d} in {refl_code}. Run gnssir first."
        )
    t = d[:, 1] + d[:, 4] / 24.0
    fit = scatter_about_spline(t, d[:, 2], d[:, 12], 3.0, True)
    out = {
        "sta": sta,
        "cfg": cfg,
        "spec": spec,
        "arcs_per_day": len(d) / max(1, len(np.unique(d[:, 1]))),
        "spline_scatter": fit["robust_scatter_m"],
        "rh": (float(np.percentile(d[:, 2], 1)), float(np.percentile(d[:, 2], 99))),
    }
    gauge = (
        read_gauge(spec["gauge_csv"], spec["gauge_time"], spec["gauge_level"], year)
        if spec.get("gauge_csv")
        else None
    )
    inv = pd.read_csv(spec["inverse_csv"]) if spec.get("inverse_csv") else None
    if gauge:
        out["slope"], out["arc_scatter"] = gauge_fit(t, d[:, 2], d[:, 12], gauge)
        if inv is not None:
            inv = inv[
                (inv.doy_fraction >= gauge[0].min()) & (inv.doy_fraction <= gauge[0].max())
            ].reset_index(drop=True)
            wl = np.interp(inv.doy_fraction, *gauge)
            A = np.c_[-inv.rh_m.values, np.ones(len(inv))]
            c = np.linalg.lstsq(A, wl, rcond=None)[0]
            out["inv_rms"], out["inv_slope"] = float(np.sqrt(np.mean((wl - A @ c) ** 2))), float(
                c[0]
            )
    # figures
    out["circle"] = str(Path(fig_dir, f"{sta}_circle.png"))
    make_figure(sta, year, top["circle_doy"], out["circle"], top.get("circle_freqs"))
    s1, s2 = top["series_days"]
    out["series"] = str(Path(fig_dir, f"{sta}_series.png"))
    series_figure(
        out["series"],
        s1,
        s2,
        t,
        fit["rh_corrected"],
        None if inv is None else inv.doy_fraction.values,
        None if inv is None else inv.rh_m.values,
        gauge[0] if gauge else None,
        gauge[1] if gauge else None,
        spec.get("gauge_name", "gauge").split(",")[0],
        year,
    )
    if gauge and spec.get("open_extension"):
        q = load_results(refl_code, sta, year, d1, d2, spec["open_extension"])
        limits = np.unique(
            np.r_[np.round(np.arange(0, 6.01, 0.2), 1), cfg["PkNoise"]]
        )  # the limit in use is always a row
        rows = tuning_table(
            q[:, 1] + q[:, 4] / 24.0, q[:, 2], q[:, 13], q[:, 12], gauge[0], gauge[1], limits
        )
        out["tuning_png"] = str(Path(fig_dir, f"{sta}_tuning.png"))
        tuning_figure(out["tuning_png"], rows, cfg["PkNoise"])
        out["tuning"] = [
            r for r in rows if r["limit"] == 0.0 or abs(r["limit"] - cfg["PkNoise"]) < 1e-9
        ]
    if gauge and spec.get("wind_csv"):
        wind = read_gauge(
            spec["wind_csv"], spec.get("wind_time", "time"), spec.get("wind_speed", "speed"), year
        )
        out["wind"] = wind_table(
            t, d[:, 2], d[:, 13], d[:, 12], gauge[0], gauge[1], wind[0], wind[1], freq=d[:, 10]
        )
        out["wind_png"] = str(Path(fig_dir, f"{sta}_wind.png"))
        if not wind_figure(out["wind_png"], out["wind"]):
            del out["wind"], out["wind_png"]  # no wind bin has enough arcs: leave the block out
    # checks
    snr = sorted(glob.glob(f"{refl_code}/{year}/snr/{sta}/{sta}{d1:03d}0.{year % 100:02d}.snr*"))
    if snr:
        try:
            out["sampling"] = sampling_check(
                read_snr(snr[0]),
                cfg["e1"],
                cfg["e2"],
                cfg.get("azval2", [0, 360]),
                cfg["maxH"],
                cfg["freqs"],
            )
            if not out["sampling"]["signals"]:
                del out["sampling"]
        except ValueError:
            pass
    if spec.get("rinex") and os.path.exists(spec["rinex"]):
        out["l2"] = l2_signal_check(spec["rinex"])
    return out


def _img(path, alt, wide=False):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<div class="figbox{" wide" if wide else ""}"><img alt="{html.escape(alt)}" src="data:image/png;base64,{b}"></div>'


def _az(a):
    return ", ".join(f"{x:g}° to {y:g}°" for x, y in zip(a[0::2], a[1::2]))


def render(blocks, top):
    def e(value):
        return html.escape(str(value))

    rows, secs = "", ""
    for b in blocks:
        sp, cfg, sta = b["spec"], b["cfg"], e(b["sta"])
        has_g = "arc_scatter" in b
        rows += (
            f'<tr><th scope="row"><a href="#{sta}">{sta.upper()}</a><br><span class="sub">{e(sp.get("place", ""))}</span></th>'
            f'<td class="n">{b["arcs_per_day"]:.0f}</td><td class="n">{b["arc_scatter"]:.3f} m</td>'
            if has_g
            else f'<tr><th scope="row"><a href="#{sta}">{sta.upper()}</a><br><span class="sub">{e(sp.get("place", ""))}</span></th>'
            f'<td class="n">{b["arcs_per_day"]:.0f}</td><td class="n">no gauge</td>'
        )
        rows += (
            f'<td class="n">{b["spline_scatter"]:.3f} m</td>'
            f'<td class="n">{b["inv_rms"]:.3f} m</td>'
            if "inv_rms" in b
            else f'<td class="n">{b["spline_scatter"]:.3f} m</td><td class="n">{"no gauge" if not has_g else "not run"}</td>'
        )
        rows += f'<td class="n">{b["slope"]:.3f}</td>' if has_g else '<td class="n">no gauge</td>'
        rows += f'<td>{e(sp.get("gauge_name", "none"))}</td></tr>'
        tun = ""
        if "tuning" in b and len(b["tuning"]) == 2:
            a, c = b["tuning"]
            tun = (
                f'<h3>Peak-to-noise limit</h3><div class="pair"><figure>{_img(b["tuning_png"], "Scatter against the gauge and arcs per day against the peak-to-noise limit")}'
                f'<figcaption>Days {top["doy"]:03d} to {top["doy_end"]:03d}, processed with the amplitude and peak-to-noise limits off. Dashed line: the limit in the station file.</figcaption></figure>'
                '<div class="tw"><table><thead><tr><th>Limit</th><th>Arcs per day</th><th>Scatter</th><th>Slope</th></tr></thead><tbody>'
                f'<tr><td>off</td><td class="n">{a["per_day"]:.0f}</td><td class="n">{a["scatter_m"]:.3f} m</td><td class="n">{a["slope"]:.3f}</td></tr>'
                f'<tr><td>{cfg["PkNoise"]:g} (the limit in use; amplitude limit still off)</td><td class="n">{c["per_day"]:.0f}</td><td class="n">{c["scatter_m"]:.3f} m</td><td class="n">{c["slope"]:.3f}</td></tr>'
                "</tbody></table></div></div>"
            )
        wnd = ""
        if "wind" in b:
            trs = "".join(
                (
                    f'<tr><td>{r["label"]} m/s</td><td class="n">{r["n"]}</td><td class="n">{r["scatter_m"]:.3f} m</td>'
                    f'<td class="n">{100 * r["outliers"]:.1f}%</td><td class="n">{r["pk"]:.1f}</td></tr>'
                    if r["scatter_m"] is not None
                    else f'<tr><td>{r["label"]} m/s</td><td class="n">{r["n"]}</td><td colspan="3">too few arcs</td></tr>'
                )
                for r in b["wind"]
            )
            wnd = (
                f'<h3>Wind</h3><div class="pair"><figure>{_img(b["wind_png"], "Scatter of single heights against the gauge, and share of arcs, by wind speed")}'
                f'<figcaption>Days {top["doy"]:03d} to {top["doy_end"]:03d}. Wind: {e(sp.get("wind_name", "wind sensor"))}, at the time of each arc. '
                "Scatter of single heights against the gauge, from one fit over all arcs with one offset per signal.</figcaption></figure>"
                '<div class="tw"><table><thead><tr><th>Wind</th><th>Arcs</th><th>Scatter</th><th>Above 0.30 m</th><th>Peak-to-noise</th></tr></thead>'
                f"<tbody>{trs}</tbody></table></div></div>"
            )
        checks = ""
        if "sampling" in b:
            s = min(b["sampling"]["signals"].values(), key=lambda v: v["wavelength_m"])
            checks += (
                f'<li><b>Sampling.</b> {b["sampling"]["interval_s"]:.0f} s. It resolves heights to {s["resolvable_height_m"]:.0f} m; '
                f'the height window ends at {cfg["maxH"]:g} m. Slower sampling can still lose arcs to the limits.</li>'
            )
        if "l2" in b:
            l2 = b["l2"]
            codes = ", ".join(f"{e(k)} on {v} satellites" for k, v in l2["satellites"].items() if v)
            checks += f'<li><b>L2 signal in the file.</b> {e(L2_VERDICT_TEXT[l2["verdict"]])} ({codes}; first {l2["epochs_read"]} epochs.)</li>'
        checks += f'<li><b>Scatter of single heights about a smooth curve (no gauge needed).</b> {b["spline_scatter"]:.3f} m.</li>'
        checks += "".join(f"<li>{e(n)}</li>" for n in sp.get("notes", []))
        series_cap = (
            "Grey points: one height per satellite arc, with the height-rate correction."
            + (
                " Blue: inverse method; the line is broken where too few arcs support it."
                if sp.get("inverse_csv")
                else ""
            )
            + (
                " Dark line: gauge. Each series is shifted by one constant to the gauge; no scale is fitted."
                if has_g
                else " No gauge exists; levels are about the median."
            )
        )
        secs += f"""<section id="{sta}">
<h2>{sta.upper()} <span class="place">{e(sp.get("place", ""))}</span></h2>
<p class="role">{e(sp.get("role", ""))}</p>
<dl class="facts">
<div><dt>Elevation</dt><dd>{cfg["e1"]:g}° to {cfg["e2"]:g}°</dd></div>
<div><dt>Height window</dt><dd>{cfg["minH"]:g} to {cfg["maxH"]:g} m</dd></div>
<div><dt>Heights found</dt><dd>{b["rh"][0]:.1f} to {b["rh"][1]:.1f} m</dd></div>
<div><dt>Azimuth</dt><dd>{_az(cfg.get("azval2", [0, 360]))}</dd></div>
<div><dt>Limits</dt><dd>amplitude {cfg["reqAmp"][0]:g}, peak-to-noise {cfg["PkNoise"]:g}</dd></div>
<div><dt>Signals</dt><dd>{len(cfg["freqs"])}</dd></div>
<div><dt>Receiver</dt><dd>{e(sp.get("receiver", "not stated"))}</dd></div>
<div><dt>Gauge</dt><dd>{e(sp.get("gauge_name", "none"))}</dd></div>
</dl>
<h3>Where the reflections come from</h3>
<figure>{_img(b["circle"], "Site image with reflection zones, next to circular periodogram plots", True)}
<figcaption>{CIRCLE_CAPTION.format(doy=top["circle_doy"])} Scroll sideways on a narrow screen.</figcaption></figure>
<h3>Water level</h3>
<figure>{_img(b["series"], "Water level from single arcs" + (", the inverse method" if sp.get("inverse_csv") else "") + (" and the gauge" if has_g else ""))}
<figcaption>Days {top["series_days"][0]:03d} to {top["series_days"][1]:03d}. {series_cap}</figcaption></figure>
{tun}
{wnd}
<h3>Station checks</h3>
<ul>{checks}</ul>
</section>
"""
    title = e(top.get("title", "GNSS-IR Station Results"))
    return f"""<title>{title}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<main>
<header style="display:flex;flex-direction:column;gap:10px">
<h1>{title}</h1>
<p class="lede">Water level from reflected GNSS signals. Each station has the same blocks, where the data for them exist: where the reflections come from, the water level series, the effect of the peak-to-noise limit, and the station checks.</p>
<p class="meta">Built {datetime.date.today().isoformat()} by <code>scripts/station_page.py</code> from gnssrefl results for {top["year"]}, days {top["doy"]:03d} to {top["doy_end"]:03d}.</p>
</header>
<div class="tw"><table>
<thead><tr><th>Station</th><th>Arcs per day</th><th>Scatter of one height against the gauge</th><th>Scatter about a smooth curve</th><th>Inverse method against the gauge (RMS)</th><th>Slope against the gauge</th><th>Gauge</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="meta">Scatter is robust (1.4826 times the median absolute deviation). One height is the result of one satellite arc. The inverse method fits one smooth height curve to the signal of all arcs and gives 5-minute values. A slope of 1 means no scale error.</p>
{secs}
</main>
"""


def main(argv=None):
    import yaml

    p = argparse.ArgumentParser(description="Build a static results page for a list of stations.")
    p.add_argument("--spec", required=True, help="YAML file; see the module text")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--fragment",
        action="store_true",
        help="write the page body only, without the html and body tags",
    )
    a = p.parse_args(argv)
    rc = os.environ.get("REFL_CODE")
    if not rc:
        raise SystemExit("Set REFL_CODE to the gnssrefl workspace.")
    top = yaml.safe_load(Path(a.spec).read_text())
    top.setdefault("year", datetime.date.today().year)
    top.setdefault("doy_end", top["doy"])
    top.setdefault("circle_doy", top["doy"])
    top.setdefault("series_days", [top["doy"], min(top["doy"] + 7, top["doy_end"])])
    fig_dir = Path(a.out).with_suffix("")
    fig_dir = fig_dir.parent / (fig_dir.name + "_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        station_block(sta.lower(), spec or {}, top, rc, fig_dir)
        for sta, spec in top["stations"].items()
    ]
    body = render(blocks, top)
    if not a.fragment:
        body = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'
            + body
            + "</body></html>"
        )
    Path(a.out).write_text(body, encoding="utf8")
    print(f"Written {a.out} ({len(body) / 1e6:.1f} MB); figures in {fig_dir}")


if __name__ == "__main__":
    main()
