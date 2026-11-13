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


def inside(t, ref_t):
    """True where a time lies inside the time span of a reference series (gauge, wind).
    Outside it an interpolation would repeat the first or the last value."""
    ref_t = np.asarray(ref_t, float)
    t = np.asarray(t, float)
    return (t >= ref_t.min()) & (t <= ref_t.max()) if len(ref_t) else np.zeros(len(t), bool)


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
        a_in = inside(arc_t, gauge_t)
        off_arc = np.median(np.interp(arc_t[a_in], gauge_t, gauge_wl) + arc_rh[a_in])
        if has_inv:
            i_in = inside(inv_t, gauge_t)
            inv_t, inv_rh = inv_t[i_in], inv_rh[i_in]
            off_inv = np.median(np.interp(inv_t, gauge_t, gauge_wl) + inv_rh)
        else:
            off_inv = None
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
    ok = inside(t, gauge_t) & np.isfinite(rh) & np.isfinite(pk) & np.isfinite(edotf)
    t, rh, pk, edotf = t[ok], rh[ok], pk[ok], edotf[ok]
    wl = np.interp(t, gauge_t, gauge_wl)
    rate = (np.interp(t + 300 / 86400, gauge_t, gauge_wl) - np.interp(t - 300 / 86400, gauge_t, gauge_wl)) / 600 * 3600
    days = max(1, len(np.unique(np.floor(t))))
    limits = np.arange(0, 6.01, 0.2) if limits is None else limits
    rows = []
    for lim in limits:
        m = pk > lim
        row = {"limit": float(lim), "n": int(m.sum()), "kept": float(m.mean()) if len(m) else 0.0, "per_day": m.sum() / days,
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


WIND_BINS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 99))


def wind_table(t, rh, pk, edotf, gauge_t, gauge_wl, wind_t, wind_speed, bins=WIND_BINS, min_arcs=100):
    """Height quality against wind speed at the time of each arc.

    One fit of (-RH) on the gauge with the height-rate term gives the residuals. Returns a
    list of dicts per wind bin: label, n, share, scatter_m (robust), outliers (share of
    residuals above 0.30 m) and pk (median peak-to-noise); bins with fewer than `min_arcs`
    arcs have scatter_m = None.
    """
    t, rh, pk, edotf = (np.asarray(v, float) for v in (t, rh, pk, edotf))
    ok = inside(t, gauge_t) & inside(t, wind_t) & np.isfinite(rh) & np.isfinite(pk) & np.isfinite(edotf)
    t, rh, pk, edotf = t[ok], rh[ok], pk[ok], edotf[ok]
    if len(t) < 3:
        return []
    wl = np.interp(t, gauge_t, gauge_wl)
    rate = (np.interp(t + 300 / 86400, gauge_t, gauge_wl) - np.interp(t - 300 / 86400, gauge_t, gauge_wl)) / 600 * 3600
    A = np.c_[wl, edotf * rate, np.ones(len(t))]
    res = -rh - A @ np.linalg.lstsq(A, -rh, rcond=None)[0]
    res = res - np.median(res)
    wind = np.interp(t, wind_t, wind_speed)
    rows = []
    for lo, hi in bins:
        m = (wind >= lo) & (wind < hi)
        row = {"label": f"{lo:g} to {hi:g}" if hi < 90 else f"above {lo:g}", "lo": lo, "n": int(m.sum()),
               "share": float(m.mean()), "scatter_m": None, "outliers": None, "pk": None}
        if m.sum() >= min_arcs:
            r = res[m]
            row.update(scatter_m=float(1.4826 * np.median(np.abs(r - np.median(r)))),
                       outliers=float((np.abs(r) > 0.30).mean()), pk=float(np.median(pk[m])))
        rows.append(row)
    return rows


def wind_figure(out, rows):
    """Scatter of single heights against the gauge, and share of arcs, per wind bin.
    Returns False, with no figure, when no bin has enough arcs."""
    ok = [r for r in rows if r["scatter_m"] is not None]
    if not ok:
        return False
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot([rows.index(r) for r in ok], [r["scatter_m"] for r in ok], "o-", color=BLUE, lw=2, ms=6)
    for r in ok:
        a1.annotate(f"{r['scatter_m']:.3f}", (rows.index(r), r["scatter_m"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=INK)
    a1.set_ylabel("scatter against\nthe gauge (m)", color=INK, fontsize=9)
    a1.set_ylim(0, 1.35 * max(r["scatter_m"] for r in ok))
    a2.bar(x, [100 * r["share"] for r in rows], color=GREY, width=0.6)
    a2.set_ylabel("share of\narcs (%)", color=INK, fontsize=9)
    a2.set_xticks(x)
    a2.set_xticklabels([r["label"] for r in rows])
    a2.set_xlabel("wind speed at the time of the arc (m/s)", color=INK)
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True

