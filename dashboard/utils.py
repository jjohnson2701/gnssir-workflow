# ABOUTME: Dashboard constants, color schemes, and utility functions
# ABOUTME: Shared across all dashboard modules — no external dependencies beyond stdlib

"""Dashboard constants and utilities."""

from pathlib import Path

# Project root (parent of dashboard/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Dark theme ---
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
)
DARK_BG = "#0d1117"
DARK_CARD = "#161b22"
DARK_BORDER = "#30363d"
DARK_TEXT = "#c9d1d9"

# --- Frequency colors ---
FREQ_COLORS = {
    "L1": "#1f77b4", "L2C": "#ff7f0e", "L5": "#2ca02c",
    "E6": "#d62728", "B3": "#9467bd", "OTHER": "#8c564b",
}

# --- State colors and ordering ---
V3_COLORS = {
    "open_water": "#2d9a6b",
    "freeze_up": "#f0ad4e",
    "ice_surface": "#4a90d9",
    "ice_layered": "#7b4fbf",
    "ice_decaying": "#e07b39",
    "break_up": "#d94452",
    # Discovery mode
    "baseline": "#2d9a6b",
    "anomalous": "#e07b39",
    "regime_change": "#f0ad4e",
}

V3_ORDER = ["open_water", "freeze_up", "ice_surface", "ice_layered", "ice_decaying", "break_up"]
DISCOVERY_ORDER = ["baseline", "regime_change", "anomalous"]

ICE_STATES_SET = {"freeze_up", "ice_surface", "ice_layered", "ice_decaying", "break_up"}
ANOMALOUS_STATES_SET = {"anomalous", "regime_change"}

# --- Tab styling ---
TAB_STYLE = {"backgroundColor": "#161b22", "color": "#8b949e", "border": "1px solid #30363d"}
TAB_SELECTED = {"backgroundColor": "#21262d", "color": "#e0e0e0", "borderTop": "2px solid #4a90d9"}
SUBTAB_STYLE = {"backgroundColor": "#161b22", "color": "#8b949e",
                "border": "1px solid #30363d", "padding": "6px 12px"}
SUBTAB_SELECTED = {"backgroundColor": "#21262d", "color": "#e0e0e0",
                   "borderTop": "2px solid #58a6ff", "padding": "6px 12px"}

# --- Feature metadata ---
FEATURE_LABELS = {
    "amp_mean": "Amplitude", "gamma_med": "Damping γ", "clr_med": "CLR",
    "af_med": "Area Factor", "pr_med": "Peak Ratio", "rh_std": "RH Std",
    "vs_med": "SNR Variance", "ms_med": "Mean Spectral Power",
    "gamma_r2_med": "Surface Coherence (γ_r2)",
    "clr_z": "CLR (z-scored)", "af_z": "AF (z-scored)", "pr_z": "PR (z-scored)",
    "gamma_z": "γ (z-scored)", "ms_z": "MS (z-scored)", "vs_z": "VS (z-scored)",
    "clr_wmed": "CLR (PRN-weighted)", "af_wmed": "AF (PRN-weighted)",
    "pr_wmed": "PR (PRN-weighted)", "gamma_wmed": "γ (PRN-weighted)",
    "ms_wmed": "MS (PRN-weighted)", "vs_wmed": "VS (PRN-weighted)",
    "amp_ratio_L1_L5": "Amp ratio L1/L5", "amp_ratio_L1_L2C": "Amp ratio L1/L2C",
    "delta_rh_mean": "ΔRH mean", "phase_circ_std": "Phase circ. std",
    "n_arcs": "Arc count", "frac_full_arc": "Full-arc fraction",
}

BEST_VARIANT = {
    "AF": "af_wmed", "MS": "ms_z", "VS": "vs_wmed",
    "CLR": "clr_wmed", "PR": "pr_wmed", "gamma": "gamma_med",
    "gamma_r2": "gamma_r2_med", "amp": "amp_mean", "rh_std": "rh_std",
}

MAJORITY_DIRECTION = {
    "amp_mean": "winter_high", "gamma_med": "summer_high", "clr_med": "summer_high",
    "af_med": "winter_high", "pr_med": "winter_high", "rh_std": "summer_high",
    "vs_med": "winter_high",
}

# Feature definitions are imported from the legacy module to avoid
# duplicating the 100+ line dictionary
INVESTIGATION_IMG_DIR = PROJECT_ROOT / "docs" / "images" / "investigation"


# --- Helper functions ---

def get_mode(v3):
    """Detect classification mode from v3 DataFrame."""
    if v3 is None:
        return "validated"
    if "classification_mode" in v3.columns:
        return v3["classification_mode"].iloc[0]
    if v3["v3_state"].isin(["baseline", "anomalous", "regime_change"]).any():
        return "discovery"
    return "validated"


def state_order(mode):
    """Return state sequence for the given mode."""
    return DISCOVERY_ORDER if mode == "discovery" else V3_ORDER


def non_baseline_states(mode):
    """Return the set of non-baseline states for the given mode."""
    return ANOMALOUS_STATES_SET if mode == "discovery" else ICE_STATES_SET


def cohens_d(group_a, group_b):
    """Compute Cohen's d (effect size) between two groups."""
    import numpy as np
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return np.nan
    ma, mb = group_a.mean(), group_b.mean()
    sa, sb = group_a.std(ddof=1), group_b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((mb - ma) / pooled)
