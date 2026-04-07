# ABOUTME: Hierarchical clustering of SNR features with dendrogram + clustered heatmap.
# ABOUTME: Shows which features are redundant vs independent, optionally split by surface state.

"""
Hierarchical clustering of SNR features to reveal redundancy structure.

Complements compare_features_pca.py:
  - PCA shows which linear combinations explain variance
  - Dendrogram shows which features measure the same thing (cluster together)
  - Merge height = redundancy: low merge = nearly identical information

Produces:
  1. Clustered correlation heatmap with dendrograms on both axes (static PNG)
  2. Plotly-native dendrogram + heatmap (interactive HTML, embeddable in Dash)
  3. Per-state dendrograms if ice classification is available (shows whether
     feature relationships change across surface states)
  4. Cophenetic correlation coefficient (how well the tree preserves distances)
  5. Feature clusters at user-specified distance threshold (CSV)

Usage:
    python scripts/compare_features_dendrogram.py --station ROSS --year 2024
    python scripts/compare_features_dendrogram.py --station ROSS --year 2024 --split-by-state
    python scripts/compare_features_dendrogram.py --station ROSS --year 2024 --threshold 0.5
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler

# Reuse the same feature list and load pattern as compare_features_pca.py
# Note: snr_features.parquet has "phase" (radians), not "phase_deg"/"phase_amp"
FEATURE_COLS = ["CLR", "PR", "AF", "gamma", "phase", "MS", "VS", "SP"]

# Extended feature list when ERA5/TEC context is merged
CONTEXT_COLS = ["t2m", "wind_speed", "tec_vtec", "precip"]

# v3 state colors (superset of v1 ice/water/transition)
STATE_COLORS = {
    "open_water": "#2d9a6b",
    "ice_surface": "#4a90d9",
    "ice_layered": "#7b4fbf",
    "ice_decaying": "#e07b39",
    "transition": "#f0ad4e",
    "unknown": "#999999",
    # v1 fallback
    "water": "#2d9a6b",
    "ice": "#4a90d9",
}

# Linkage method — Ward minimizes within-cluster variance, good default.
# 'average' (UPGMA) is more common for correlation-distance clustering.
DEFAULT_LINKAGE = "average"
DEFAULT_METRIC = "correlation"  # 1 - Pearson r

# Cluster colors — each cluster gets a distinct color in the dendrogram.
# Branch color = the cluster it leads to; above-threshold branches stay gray.
CLUSTER_PALETTE = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
]


def load_data(station: str, year: int, results_dir: Path):
    """Load SNR features and optionally ice classification + context variables."""
    snr_path = results_dir / station / f"{station}_{year}_snr_features.parquet"
    # Prefer v3 classification, fall back to v1
    ice_v3_path = results_dir / station / f"{station}_{year}_ice_classification_v3.parquet"
    ice_v1_path = results_dir / station / f"{station}_{year}_ice_classification.parquet"

    if not snr_path.exists():
        raise FileNotFoundError(f"SNR features not found: {snr_path}")

    snr = pd.read_parquet(snr_path)
    print(f"Loaded {len(snr)} SNR feature rows from {snr_path.name}")

    # Check which features are available
    available = [c for c in FEATURE_COLS if c in snr.columns]
    context = [c for c in CONTEXT_COLS if c in snr.columns]
    if context:
        print(f"  Context variables found: {context}")
        available += context
    missing = [c for c in FEATURE_COLS if c not in snr.columns]
    if missing:
        print(f"  Missing features (will skip): {missing}")
    print(f"  Available for clustering: {available}")

    # Ensure doy column
    if "doy" not in snr.columns and "date" in snr.columns:
        snr["doy"] = pd.to_datetime(snr["date"]).dt.dayofyear

    # Load classification labels — prefer v3 over v1
    labels = None

    if ice_v3_path.exists():
        ice = pd.read_parquet(ice_v3_path)
        print(f"Loaded {len(ice)} v3 classification rows from {ice_v3_path.name}")
        if "v3_state" in ice.columns:
            # v3 is daily — merge on doy
            if "doy" not in ice.columns and "date" in ice.columns:
                ice["doy"] = pd.to_datetime(ice["date"]).dt.dayofyear
            ice_merge = ice[["doy", "v3_state"]].rename(columns={"v3_state": "state"})
            snr = snr.merge(ice_merge, on="doy", how="left")
            labels = snr["state"].fillna("unknown")
            print(f"  State distribution: {labels.value_counts().to_dict()}")

    elif ice_v1_path.exists():
        ice = pd.read_parquet(ice_v1_path)
        print(f"Loaded {len(ice)} v1 classification rows from {ice_v1_path.name}")
        if "classification" in ice.columns:
            if "doy" not in ice.columns and "date" in ice.columns:
                ice["doy"] = pd.to_datetime(ice["date"]).dt.dayofyear
            ice_merge = ice[["doy", "classification"]].rename(
                columns={"classification": "state"}
            )
            snr = snr.merge(ice_merge, on="doy", how="left")
            labels = snr["state"].fillna("unknown")
            print(f"  State distribution: {labels.value_counts().to_dict()}")
    else:
        print(f"  No classification found")

    return snr, labels, available


def _cluster_colors_for_dendrogram(Z, threshold, n_leaves):
    """Map each internal node in the linkage to its cluster color.

    scipy's dendrogram draws n-1 U-shapes, one per merge. We assign each
    U-shape a color: if the merge height is below the threshold, the branch
    belongs to a specific cluster and gets that cluster's color. Merges above
    the threshold get gray.

    Returns a dict mapping scipy's default color keys to our palette colors,
    suitable for passing as `link_color_func` to dendrogram().
    """
    from scipy.cluster.hierarchy import fcluster, leaders

    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    n_merges = len(Z)

    # Build a mapping: node_id -> cluster_id
    # Leaf nodes: id = 0..n_leaves-1, cluster from fcluster
    # Internal nodes: id = n_leaves..n_leaves+n_merges-1
    node_cluster = {}
    for i in range(n_leaves):
        node_cluster[i] = cluster_ids[i]

    for i in range(n_merges):
        left = int(Z[i, 0])
        right = int(Z[i, 1])
        merge_height = Z[i, 2]
        node_id = n_leaves + i

        left_c = node_cluster.get(left)
        right_c = node_cluster.get(right)

        if merge_height <= threshold and left_c == right_c and left_c is not None:
            node_cluster[node_id] = left_c
        else:
            node_cluster[node_id] = None  # above threshold or mixed

    # Build color function: maps node_id -> color string
    unique_clusters = sorted(set(c for c in cluster_ids))
    cluster_color_map = {}
    for idx, cid in enumerate(unique_clusters):
        cluster_color_map[cid] = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]

    def link_color_func(node_id):
        c = node_cluster.get(node_id)
        if c is not None:
            return cluster_color_map[c]
        return "#888888"

    return link_color_func, cluster_color_map, unique_clusters


def compute_correlation_distance(df: pd.DataFrame, feature_cols: list, metric="correlation"):
    """Compute distance matrix between features.

    Args:
        df: DataFrame with feature columns
        feature_cols: list of column names
        metric: "correlation" (1 - |Pearson r|) or "partial" (1 - |partial r|)
            Partial correlation uses the precision matrix (inverse covariance),
            which is the same structure underlying Mahalanobis distance.
            It shows direct relationships between features after removing
            the effect of all other features.

    Returns:
        (corr_or_pcorr, dist) where corr_or_pcorr is the similarity matrix
        and dist is the distance matrix.
    """
    if metric == "partial":
        from sklearn.preprocessing import StandardScaler
        scaled = StandardScaler().fit_transform(df[feature_cols])
        cov = np.cov(scaled, rowvar=False)
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.inv(cov + 1e-6 * np.eye(len(feature_cols)))

        # pcorr(i,j) = -P(i,j) / sqrt(P(i,i)*P(j,j))
        d = np.sqrt(np.diag(precision))
        pcorr = -precision / np.outer(d, d)
        np.fill_diagonal(pcorr, 1.0)
        corr = pd.DataFrame(pcorr, index=feature_cols, columns=feature_cols)
    else:
        corr = df[feature_cols].corr()

    # 1 - |r|: 0 = identical information, 1 = completely independent
    dist = 1 - corr.abs()
    # Ensure diagonal is exactly 0 and matrix is symmetric
    np.fill_diagonal(dist.values, 0)
    dist = (dist + dist.T) / 2
    return corr, dist


def build_clustered_heatmap_mpl(corr, dist, feature_cols, output_path, title="",
                                threshold=0.4):
    """Build a matplotlib figure: colored dendrogram on both axes + correlation heatmap.

    Branch colors indicate cluster membership at the given threshold.
    Branches above the threshold are gray (inter-cluster merges).
    """
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=DEFAULT_LINKAGE)

    # Cophenetic correlation — how well the tree preserves pairwise distances
    coph_r, _ = cophenet(Z, condensed)

    # Cluster color mapping
    link_color_func, cluster_color_map, unique_clusters = \
        _cluster_colors_for_dendrogram(Z, threshold, len(feature_cols))

    fig = plt.figure(figsize=(11, 9.5))

    # Layout: dendrogram top, dendrogram left, heatmap center, colorbar right
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.15, 0.75, 0.04],
        height_ratios=[0.20, 0.75],
        wspace=0.02, hspace=0.02,
    )

    # Top dendrogram — colored by cluster
    ax_top = fig.add_subplot(gs[0, 1])
    dend_top = dendrogram(Z, labels=feature_cols, ax=ax_top,
                          no_labels=True,
                          link_color_func=link_color_func)
    ax_top.set_xticks([])
    ax_top.spines[:].set_visible(False)
    ax_top.set_ylabel("Distance\n(1-|r|)", fontsize=8)
    # Threshold line
    ax_top.axhline(y=threshold, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax_top.text(ax_top.get_xlim()[1] * 0.98, threshold + 0.01,
                f"t={threshold}", ha="right", fontsize=7, color="#888888")

    # Reorder features by dendrogram leaf order
    order = dend_top["leaves"]
    ordered_labels = [feature_cols[i] for i in order]
    corr_ordered = corr.iloc[order, :].iloc[:, order]

    # Color the leaf labels by their cluster
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    leaf_colors = [cluster_color_map[cluster_ids[i]] for i in order]

    # Left dendrogram (rotated) — colored by cluster
    ax_left = fig.add_subplot(gs[1, 0])
    dendrogram(Z, labels=feature_cols, ax=ax_left,
               orientation="left", no_labels=True,
               link_color_func=link_color_func)
    ax_left.set_yticks([])
    ax_left.spines[:].set_visible(False)
    ax_left.invert_yaxis()
    ax_left.axvline(x=threshold, color="#888888", linestyle="--", linewidth=1, alpha=0.7)

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    im = ax_heat.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1,
                        aspect="auto", interpolation="nearest")
    n = len(ordered_labels)
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=9)
    ax_heat.set_yticklabels(ordered_labels, fontsize=9)

    # Color the tick labels by cluster
    for idx, (xtick, ytick) in enumerate(zip(ax_heat.get_xticklabels(),
                                              ax_heat.get_yticklabels())):
        xtick.set_color(leaf_colors[idx])
        xtick.set_fontweight("bold")
        ytick.set_color(leaf_colors[idx])
        ytick.set_fontweight("bold")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_ordered.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color=color)

    # Colorbar
    ax_cb = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=ax_cb, label="Pearson r")

    # Cluster legend
    from matplotlib.patches import Patch
    legend_patches = []
    for cid in unique_clusters:
        members = [feature_cols[i] for i in range(len(feature_cols))
                    if cluster_ids[i] == cid]
        legend_patches.append(
            Patch(facecolor=cluster_color_map[cid],
                  label=f"C{cid}: {', '.join(members)}")
        )
    fig.legend(handles=legend_patches, loc="lower center", ncol=min(len(legend_patches), 3),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    suptitle = title or "Feature Clustering — Correlation Distance"
    fig.suptitle(f"{suptitle}\n(cophenetic r = {coph_r:.3f}, threshold = {threshold})",
                 fontweight="bold", fontsize=12, y=0.99)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.name}")

    return Z, order, coph_r


def build_plotly_clustergram(corr, dist, feature_cols, output_path, title="",
                             threshold=0.4):
    """Build a Plotly figure with colored dendrogram + heatmap for Dash embedding.

    Branch colors indicate cluster membership. Returns the figure object
    so it can be used directly in a Dash callback.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=DEFAULT_LINKAGE)

    # Cluster color mapping
    link_color_func, cluster_color_map, unique_clusters = \
        _cluster_colors_for_dendrogram(Z, threshold, len(feature_cols))

    # Get dendrogram structure with colors from scipy
    dend = dendrogram(Z, labels=feature_cols, no_plot=True,
                      link_color_func=link_color_func)
    order = dend["leaves"]
    ordered_labels = [feature_cols[i] for i in order]
    corr_ordered = corr.iloc[order, :].iloc[:, order]

    # Leaf cluster colors
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    leaf_colors = [cluster_color_map[cluster_ids[i]] for i in order]

    # Build figure with subplots: top dendrogram + heatmap
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.2, 0.8],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    # Top dendrogram traces — colored by cluster
    icoord = np.array(dend["icoord"])
    dcoord = np.array(dend["dcoord"])
    dend_colors = dend["color_list"]
    for i in range(len(icoord)):
        fig.add_trace(
            go.Scatter(
                x=icoord[i],
                y=dcoord[i],
                mode="lines",
                line=dict(color=dend_colors[i], width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="#888888",
                  line_width=1, row=1, col=1,
                  annotation_text=f"t={threshold}", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="#888888")

    # Heatmap
    hover_text = []
    for i, row_label in enumerate(ordered_labels):
        row_hover = []
        for j, col_label in enumerate(ordered_labels):
            val = corr_ordered.iloc[i, j]
            row_hover.append(f"{row_label} vs {col_label}<br>r = {val:.3f}")
        hover_text.append(row_hover)

    fig.add_trace(
        go.Heatmap(
            z=corr_ordered.values,
            x=ordered_labels,
            y=ordered_labels,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_ordered.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(title="Pearson r", thickness=15, len=0.6, y=0.35),
        ),
        row=2, col=1,
    )

    # Cluster legend as annotations
    legend_text = "  |  ".join(
        f"<span style='color:{cluster_color_map[cid]}'>C{cid}: "
        + ", ".join(feature_cols[i] for i in range(len(feature_cols)) if cluster_ids[i] == cid)
        + "</span>"
        for cid in unique_clusters
    )

    fig.update_layout(
        title=dict(
            text=(title or "Feature Clustering") + f"<br><sub>{legend_text}</sub>",
            font=dict(size=14),
        ),
        template="plotly_dark",
        height=750,
        width=850,
        margin=dict(l=80, r=40, t=110, b=80),
    )

    # Style dendrogram axis
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Distance (1-|r|)", showgrid=False, zeroline=False,
                     row=1, col=1)

    # Style heatmap axis
    fig.update_xaxes(tickangle=45, row=2, col=1)

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Saved: {output_path.name}")

    return fig


def build_per_state_comparison(snr, labels, feature_cols, output_dir, min_samples=50,
                                threshold=0.4, metric="correlation"):
    """Build side-by-side colored dendrograms for each surface state.

    The key question: does the feature redundancy structure change across states?
    If gamma and CLR cluster tightly during ice but separate during water, that
    tells you they carry different information depending on context.

    Branch colors show which features cluster together within each state.
    """
    states = labels.value_counts()
    viable_states = [s for s in states.index if states[s] >= min_samples and s != "unknown"]

    if len(viable_states) < 2:
        print(f"  Only {len(viable_states)} states with >={min_samples} samples, skipping per-state comparison")
        return

    print(f"\nPer-state dendrograms for: {viable_states}")

    n_states = len(viable_states)
    fig, axes = plt.subplots(1, n_states, figsize=(5 * n_states, 7))
    if n_states == 1:
        axes = [axes]

    all_orders = {}
    for ax, state in zip(axes, viable_states):
        mask = labels == state
        df_state = snr.loc[mask, feature_cols].dropna()

        if len(df_state) < min_samples:
            ax.set_title(f"{state}\n(n={len(df_state)}, too few)")
            continue

        corr_state, dist_state = compute_correlation_distance(df_state, feature_cols, metric=metric)
        condensed = squareform(dist_state.values, checks=False)
        Z_state = linkage(condensed, method=DEFAULT_LINKAGE)

        # Colored branches
        link_color_func, cluster_color_map, unique_clusters = \
            _cluster_colors_for_dendrogram(Z_state, threshold, len(feature_cols))

        dend = dendrogram(Z_state, labels=feature_cols, ax=ax,
                          leaf_rotation=90, leaf_font_size=9,
                          link_color_func=link_color_func)
        all_orders[state] = dend["leaves"]

        # Threshold line
        ax.axhline(y=threshold, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)

        # Color leaf labels
        cluster_ids = fcluster(Z_state, t=threshold, criterion="distance")
        for lbl in ax.get_xticklabels():
            feat_name = lbl.get_text()
            if feat_name in feature_cols:
                feat_idx = feature_cols.index(feat_name)
                lbl.set_color(cluster_color_map[cluster_ids[feat_idx]])
                lbl.set_fontweight("bold")

        coph_r, _ = cophenet(Z_state, condensed)
        state_color = STATE_COLORS.get(state, "#666666")
        n_clusters = len(unique_clusters)
        ax.set_title(f"{state} (n={len(df_state)})\n{n_clusters} clusters, coph r={coph_r:.3f}",
                     fontweight="bold", color=state_color)
        ax.set_ylabel("Distance (1-|r|)")

    fig.suptitle("Feature Clustering by Surface State", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "dendrogram_per_state.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: dendrogram_per_state.png")

    # Report whether leaf ordering changed between states
    if len(all_orders) >= 2:
        states_list = list(all_orders.keys())
        print(f"\n  Leaf ordering comparison:")
        for i, s1 in enumerate(states_list):
            for s2 in states_list[i+1:]:
                same = all_orders[s1] == all_orders[s2]
                print(f"    {s1} vs {s2}: {'SAME' if same else 'DIFFERENT'} ordering")


def extract_clusters(Z, feature_cols, threshold, output_path):
    """Cut the dendrogram at a given distance threshold and report clusters."""
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    clusters = pd.DataFrame({
        "feature": feature_cols,
        "cluster": cluster_ids,
    }).sort_values("cluster")

    print(f"\nFeature clusters at distance threshold {threshold}:")
    for cid in sorted(clusters["cluster"].unique()):
        members = clusters[clusters["cluster"] == cid]["feature"].tolist()
        print(f"  Cluster {cid}: {', '.join(members)}")

    clusters.to_csv(output_path, index=False)
    print(f"Saved: {output_path.name}")
    return clusters


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical clustering of SNR features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--station", required=True, help="Station code")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results_annual"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: results_annual/{station}/cluster_analysis)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Distance threshold for cluster extraction (0=identical, 1=independent)")
    parser.add_argument("--split-by-state", action="store_true",
                        help="Generate per-surface-state dendrograms")
    parser.add_argument("--metric", default="correlation",
                        choices=["correlation", "partial"],
                        help="Distance metric: 'correlation' (1-|r|) or 'partial' "
                             "(1-|partial r| from precision matrix, same structure as Mahalanobis)")
    parser.add_argument("--linkage-method", default=DEFAULT_LINKAGE,
                        choices=["average", "ward", "complete", "single"],
                        help="Linkage method for hierarchical clustering")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Minimum samples per state for per-state analysis")
    args = parser.parse_args()

    output_dir = args.output or (args.results_dir / args.station / "cluster_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    snr, labels, available = load_data(args.station, args.year, args.results_dir)

    # Drop NaN rows
    df = snr[available].dropna()
    n_dropped = len(snr) - len(df)
    print(f"\nClustering input: {len(df)} arcs ({n_dropped} dropped for NaN)")

    if len(df) < 100:
        print("ERROR: Too few valid arcs for meaningful clustering")
        return

    # Compute correlation distance
    corr, dist = compute_correlation_distance(df, available, metric=args.metric)
    metric_label = "Partial Correlation" if args.metric == "partial" else "Correlation"
    print(f"  Distance metric: {metric_label}")

    # 1. Static clustered heatmap (matplotlib)
    Z, order, coph_r = build_clustered_heatmap_mpl(
        corr, dist, available,
        output_dir / "clustered_heatmap.png",
        title=f"{args.station} {args.year} — Feature Clustering",
        threshold=args.threshold,
    )
    print(f"  Cophenetic correlation: {coph_r:.3f} (>0.7 = good tree fit)")

    # 2. Plotly interactive version
    build_plotly_clustergram(
        corr, dist, available,
        output_dir / "clustered_heatmap.html",
        title=f"{args.station} {args.year} — Feature Clustering (1 - |Pearson r|)",
        threshold=args.threshold,
    )

    # 3. Extract clusters at threshold
    # Note: fcluster returns IDs in original feature order, not dendrogram leaf order
    extract_clusters(
        Z, available, args.threshold,
        output_dir / "feature_clusters.csv",
    )

    # 4. Per-state dendrograms
    if args.split_by_state and labels is not None:
        # Need to align labels with the NaN-dropped dataframe
        labels_aligned = labels.loc[df.index]
        build_per_state_comparison(
            snr.loc[df.index], labels_aligned, available, output_dir,
            min_samples=args.min_samples,
            threshold=args.threshold,
            metric=args.metric,
        )
    elif args.split_by_state:
        print("\n  --split-by-state requested but no classification labels available")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
