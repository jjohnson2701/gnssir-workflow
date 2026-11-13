# ABOUTME: Two figures for a station results page: water level series against a gauge, and the
# ABOUTME: effect of the peak-to-noise limit on scatter and on the number of arcs.
"""Figures for the station results page.

series_figure   heights of single arcs, the inverse-method curve and the gauge, with the
                difference to the gauge below when a gauge exists
tuning_figure   scatter against the gauge and arcs per day as a function of the
                peak-to-noise limit (needs a run with the limits off, and a gauge)
tuning_table    the numbers behind tuning_figure
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#22303c"      # gauge and text
BLUE = "#1b6fb5"     # inverse method
ORANGE = "#c8641a"   # limit in use
GREY = "#8a96a3"     # single arcs
GRID = "#d5dce3"


def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)


def to_level(rh, offset):
    """Water level from reflector height: a higher surface gives a smaller height."""
    return offset - np.asarray(rh, float)


def series_figure(out, day1, day2, arc_t, arc_rh, inv_t=None, inv_rh=None, gauge_t=None, gauge_wl=None,
                  gauge_name="gauge", year=None):
    """Plot days day1..day2. Times are day of year with fraction. Returns the offsets used.

    With a gauge, each height series is shifted by one constant so that its median matches
    the gauge (no scale is fitted). Without a gauge the median of the inverse series (or of
    the arcs) is set to zero.
    """
    arc_t, arc_rh = np.asarray(arc_t, float), np.asarray(arc_rh, float)
    has_inv = inv_t is not None and len(inv_t) > 0
    has_gauge = gauge_t is not None and len(gauge_t) > 0
    if has_inv:
        inv_t, inv_rh = np.asarray(inv_t, float), np.asarray(inv_rh, float)
    if has_gauge:
        gauge_t, gauge_wl = np.asarray(gauge_t, float), np.asarray(gauge_wl, float)
        off_arc = np.median(np.interp(arc_t, gauge_t, gauge_wl) + arc_rh)
        off_inv = np.median(np.interp(inv_t, gauge_t, gauge_wl) + inv_rh) if has_inv else None
    else:
        off_inv = np.median(inv_rh) if has_inv else None
        off_arc = off_inv if has_inv else np.median(arc_rh)
    rows = 2 if (has_gauge and has_inv) else 1
    fig, axes = plt.subplots(rows, 1, figsize=(11, 5.2 if rows == 2 else 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]} if rows == 2 else None)
    ax = axes[0] if rows == 2 else axes
    m = (arc_t >= day1) & (arc_t <= day2 + 1)
    ax.plot(arc_t[m], to_level(arc_rh[m], off_arc), ".", ms=3, color=GREY, alpha=0.7, label="single arcs (periodogram)", zorder=1)
    if has_gauge:
        g = (gauge_t >= day1) & (gauge_t <= day2 + 1)
        ax.plot(gauge_t[g], gauge_wl[g], color=INK, lw=2.4, label=gauge_name, zorder=2)
    if has_inv:
        k = (inv_t >= day1) & (inv_t <= day2 + 1)
        t, y = inv_t[k], to_level(inv_rh[k], off_inv)
        gap = np.where(np.diff(t) > 20.0 / 1440)[0] + 1   # break the line where output was removed
        for seg_t, seg_y in zip(np.split(t, gap), np.split(y, gap)):
            ax.plot(seg_t, seg_y, color=BLUE, lw=1.3, zorder=3)
        ax.plot([], [], color=BLUE, lw=1.3, label="inverse method")
    ax.set_ylabel("water level (m)" if has_gauge else "water level about the median (m)", color=INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=3)
    _style(ax)
    if rows == 2:
        ax2 = axes[1]
        k = (inv_t >= day1) & (inv_t <= day2 + 1)
        d = to_level(inv_rh[k], off_inv) - np.interp(inv_t[k], gauge_t, gauge_wl)
        ax2.axhline(0, color=INK, lw=0.8)
        ax2.plot(inv_t[k], 100 * d, ".", ms=2, color=BLUE)
        ax2.set_ylabel("inverse method\nminus gauge (cm)", color=INK, fontsize=9)
        _style(ax2)
    last = axes[-1] if rows == 2 else axes
    last.set_xlabel(f"day of year{' ' + str(year) if year else ''} (UTC)", color=INK)
    last.set_xlim(day1, day2 + 1)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"offset_arcs": off_arc, "offset_inverse": off_inv}


def tuning_table(t, rh, pk, edotf, gauge_t, gauge_wl, limits=None):
    """Scatter, slope and arc count against the gauge for each peak-to-noise limit.

    Returns a list of dicts: limit, n, kept, per_day, scatter_m, slope. The fit is
    (-RH) = slope * gauge + k * edotF * (gauge rate) + constant.
    """
    t, rh, pk, edotf = (np.asarray(v, float) for v in (t, rh, pk, edotf))
    wl = np.interp(t, gauge_t, gauge_wl)
    rate = (np.interp(t + 300 / 86400, gauge_t, gauge_wl) - np.interp(t - 300 / 86400, gauge_t, gauge_wl)) / 600 * 3600
    days = len(np.unique(np.floor(t)))
    limits = np.arange(0, 6.01, 0.2) if limits is None else limits
    rows = []
    for lim in limits:
        m = pk > lim
        row = {"limit": float(lim), "n": int(m.sum()), "kept": float(m.mean()), "per_day": m.sum() / days,
               "scatter_m": np.nan, "slope": np.nan}
        if m.sum() >= 30:
            A = np.c_[wl[m], edotf[m] * rate[m], np.ones(m.sum())]
            c = np.linalg.lstsq(A, -rh[m], rcond=None)[0]
            r = -rh[m] - A @ c
            row["scatter_m"] = float(1.4826 * np.median(np.abs(r - np.median(r))))
            row["slope"] = float(c[0])
        rows.append(row)
    return rows


def tuning_figure(out, rows, limit_in_use, min_kept=0.03):
    """Two stacked panels: scatter against the gauge, and arcs per day, against the limit."""
    lim = np.array([r["limit"] for r in rows])
    sd = np.array([r["scatter_m"] for r in rows])
    per_day = np.array([r["per_day"] for r in rows])
    kept = np.array([r["kept"] for r in rows])
    show = kept >= min_kept
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True)
    a1.plot(lim[show], sd[show], color=BLUE, lw=2)
    a1.set_ylabel("scatter against\nthe gauge (m)", color=INK, fontsize=9)
    a2.plot(lim, per_day, color=BLUE, lw=2)
    a2.set_ylabel("arcs per day", color=INK, fontsize=9)
    a2.set_xlabel("peak-to-noise limit", color=INK)
    a2.set_ylim(bottom=0)
    for ax in (a1, a2):
        ax.axvline(limit_in_use, color=ORANGE, lw=1.6, ls="--")
        _style(ax)
    a1.text(limit_in_use, a1.get_ylim()[1], " limit in use", color=ORANGE, fontsize=8.5, va="top", ha="left")
    a2.set_xlim(0, lim[kept > 0.005].max() + 0.2 if (kept > 0.005).any() else lim.max())
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
