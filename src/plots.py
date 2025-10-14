import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots
import seaborn as sns
from matplotlib import ticker
from numpy._typing import NDArray

# --- Default Configuration ---
DEFAULT_CSV = "../out/results.csv"
DEFAULT_OUTPUT = "subregion_analysis.png"
GROUP_ORDER = ["naive control", "rotarod control", "rotarod"]
BAR_COLORS = {
    "naive control": "#4C72B0",  # Dark Blue
    "rotarod control": "#DD8452",  # Green
    "rotarod": "#55A868",  # Orange
}


def plot_data(
    data: NDArray,
    roi: NDArray,
    thr_mask: NDArray[bool],
    thr_and_roi_mask: NDArray[bool] = None,
    particles_labels: NDArray[bool] = None,
    title: str = None,
    show_plot: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the original image, the original image with ROI, the threshold mask, and the combined threshold & ROI mask.

    :param data: Original image data as a numpy array.
    :param roi: ROI coordinates as a numpy array.
    :param thr_mask: Threshold mask as a numpy array.
    :param thr_and_roi_mask: Combined threshold and ROI mask as a numpy array.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1. Original Image
    x_shape, y_shape = data.shape
    axes[0, 0].imshow(data, cmap="seismic")
    axes[0, 0].set_title(f"Image ({x_shape:,d}x{y_shape:,d}px)")
    axes[0, 0].axis("off")

    # 2. Image with ROI
    axes[0, 1].imshow(data, cmap="seismic")
    axes[0, 1].set_title("Image with ROI")
    roi_polygon = plt.Polygon(roi, fill=None, edgecolor="r", linewidth=2)
    axes[0, 1].add_patch(roi_polygon)
    axes[0, 1].axis("off")

    # 3. Threshold Mask
    nb_pixels = thr_mask.sum()
    axes[1, 0].imshow(thr_mask, cmap="gray")
    axes[1, 0].set_title(f"Threshold Mask ({nb_pixels:,d}px)")
    axes[1, 0].axis("off")

    # 4. Combined Threshold & ROI Mask
    if thr_and_roi_mask is not None:
        nb_pixels = thr_and_roi_mask.sum()
        axes[1, 1].imshow(thr_and_roi_mask, cmap="gray")
        axes[1, 1].set_title(f"Threshold + ROI Masks ({nb_pixels:,d}px)")
        axes[1, 1].axis("off")

    # 4. Combined Threshold & ROI Mask with particles
    if particles_labels is not None:
        nb_particles = len(np.unique(particles_labels)) - 1
        axes[1, 1].imshow(particles_labels, cmap="prism", alpha=0.5)
        axes[1, 1].set_title(f"Particles ({nb_particles:,d} particles)")
        axes[1, 1].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        if isinstance(save_path, str):
            save_path = Path(save_path)

        if not save_path.suffix:
            save_path += ".png"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_data_interactive(
    data: NDArray,
    roi: NDArray,
    thr_mask: NDArray[bool],
    thr_and_roi_mask: NDArray[bool] = None,
    particles_labels: NDArray[bool] = None,
    title: str = None,
    save_path: str | Path = None,
) -> None:
    """
    Create an interactive plot of the original image, the original image with ROI,
    the threshold mask, and the combined threshold & ROI mask using Plotly.

    :param data: Original image data as a numpy array.
    :param roi: ROI coordinates as a numpy array.
    :param thr_mask: Threshold mask as a numpy array.
    :param thr_and_roi_mask: Combined threshold and ROI mask as a numpy array.
    :param particles_labels: Optional particle labels mask
    :param title: Optional title for the plot
    :param save_path: Optional path to save the plot as HTML
    """
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Image ({data.shape[0]:,d}x{data.shape[1]:,d}px)",
            "Image with ROI",
            "Threshold mask",
            "Threshold & ROI mask",
        ],
    )

    # 1. Original Image
    fig.add_trace(go.Heatmap(z=data, colorscale="RdBu", showscale=False), row=1, col=1)

    # 2. Image with ROI
    fig.add_trace(go.Heatmap(z=data, colorscale="RdBu", showscale=False), row=1, col=2)
    if roi is not None:
        fig.add_trace(
            go.Contour(
                z=roi,
                showscale=False,
                contours_coloring="lines",
                line_width=2,
                contours=dict(start=0.5, end=0.5, size=0),
            ),
            row=1,
            col=2,
        )

    # 3. Threshold mask
    fig.add_trace(
        go.Heatmap(z=thr_mask, colorscale="RdBu", showscale=False), row=2, col=1
    )

    # 4. Combined mask
    if thr_and_roi_mask is not None:
        fig.add_trace(
            go.Heatmap(z=thr_and_roi_mask, colorscale="RdBu", showscale=False),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(title=title, showlegend=False, width=1000, height=1000)

    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showticklabels=False, row=i, col=j)
            fig.update_yaxes(showticklabels=False, row=i, col=j)

    if save_path:
        fig.write_html(str(save_path))
    else:
        fig.show()


def infer_group_from_mouse_name(mouse_name):
    """
    Determine the group based on substrings in mouse_id.
    """
    lname = str(mouse_name).lower()
    if "naive control" in lname:
        return "naive control"
    elif "rotarod control" in lname:
        return "rotarod control"
    elif "rotarod" in lname:
        return "rotarod"
    else:
        return "unknown"


def load_and_prepare(csv_path):
    """
    Load a single CSV and compute mean fluorescence per mouse and subregion.
    Returns DataFrame with columns [subregion, group, mouse_id, mean_fluorescence, slice_number].
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")

    # Derive a group if missing
    if "group" not in df.columns:
        df["group"] = df["mouse_name"].apply(infer_group_from_mouse_name)

    # Extract the slice number from area_name if present
    df["slice_number"] = df["area_name"].str.extract(r"slice(\d+)", expand=False)
    df["slice_number"] = pd.to_numeric(df["slice_number"], errors="coerce")

    # Remove sliceX text from area_name
    df["area_name"] = (
        df["area_name"]
        .str.replace(r"_slice\d+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Aggregate the different slices together
    if "roi_rate" in df.columns:
        df_agg = df.groupby(["area_name", "group", "mouse_name"], as_index=False).agg(
            mean_roi_rate=("roi_rate", "mean"),
            std_roi_rate=("roi_rate", "std"),
            n_slices=("slice_number", "nunique"),
        )
    elif "particles_rate" in df.columns:
        df_agg = df.groupby(["area_name", "group", "mouse_name"], as_index=False).agg(
            mean_particles_rate=("particles_rate", "mean"),
            std_particles_rate=("particles_rate", "std"),
            n_slices=("slice_number", "nunique"),
        )
    else:
        raise ValueError("No valid columns found for aggregation.")

    return df_agg


def _prepare_plot_data(df_agg):
    """
    Prepare data for plotting by filtering subregions and determining plot type.
    Returns the filtered dataframe and plot configuration.

    :param df_agg: Aggregated DataFrame with the results
    :return: Tuple of (filtered DataFrame, plot_type, y_axis_title)
    """
    # filter subregions by keywords
    keywords = ["simplex", "crus", "paramedian", "lobule"]
    pat = "|".join(keywords)
    df_plot = df_agg[df_agg["area_name"].str.contains(pat, case=False, na=False)].copy()
    if df_plot.empty:
        print("Warning: no matching subregions, empty plot.")

    if "mean_roi_rate" in df_plot.columns and "mean_particles_rate" in df_plot.columns:
        raise ValueError(
            "Both 'roi_rate' and 'particles_rate' columns found. It should be only one."
        )

    if "mean_roi_rate" in df_plot.columns:
        plot_type = "roi_rate"
        y_axis_title = "Fluorescent area of ROI"
    elif "mean_particles_rate" in df_plot.columns:
        plot_type = "particles_rate"
        y_axis_title = "Ratio of active cells in ROI"
    else:
        raise ValueError("No valid columns found for plotting.")

    return df_plot, plot_type, y_axis_title


def plot_results(df_agg, output_path):
    """
    Plot bars + scatter, filtered and ordered, with colors.
    Y-axis scaled to percent (0–100).
    """
    df_plot, plot_type, y_axis_title = _prepare_plot_data(df_agg)

    plt.figure(figsize=(16, 9))
    g = sns.barplot(
        data=df_plot,
        x="area_name",
        y=f"mean_{plot_type}",
        hue="group",
        errorbar="se",
        palette=BAR_COLORS,
        alpha=0.6,
    )

    sns.stripplot(
        data=df_plot,
        x="area_name",
        y=f"mean_{plot_type}",
        hue="group",
        linewidth=0.5,
        edgecolor="black",
        dodge=True,
        alpha=0.6,
        ax=g,
        palette=BAR_COLORS,
    )

    # Remove extra legend handles
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[:3], labels[:3], title="", fontsize=14)

    # Configure axes
    plt.xticks(fontsize=14)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
    g.set_xlabel("", fontsize=16)
    plt.yticks(fontsize=14)

    if plot_type == "roi_rate":
        g.set_ylabel(y_axis_title, fontsize=16)
        g.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        g.set_ylim(0, 0.8)
    else:
        g.set_ylabel(y_axis_title, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()


def plot_results_interactive(df_agg, output_path: str | Path = None):
    """
    Create an interactive plot with bars + scatter, filtered and ordered, with colors using Plotly.
    Y-axis scaled to percent (0–100).

    :param df_agg: Aggregated DataFrame with the results
    :param output_path: Optional path to save the plot as HTML
    """
    df_plot, plot_type, y_axis_title = _prepare_plot_data(df_agg)

    # Create figure
    fig = go.Figure()

    # Add bars for each group
    for group in GROUP_ORDER:
        group_data = df_plot[df_plot["group"] == group]

        # Add bar plot
        fig.add_trace(
            go.Bar(
                name=group,
                x=group_data["area_name"],
                y=group_data[f"mean_{plot_type}"],
                error_y=dict(
                    type="data",
                    array=group_data[f"std_{plot_type}"]
                    / np.sqrt(group_data["n_slices"]),
                    visible=True,
                ),
                marker_color=BAR_COLORS[group],
                opacity=0.6,
            )
        )

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                name=group,
                x=group_data["area_name"],
                y=group_data[f"mean_{plot_type}"],
                mode="markers",
                marker=dict(
                    color=BAR_COLORS[group], size=8, line=dict(color="black", width=1)
                ),
                text=group_data["mouse_name"],  # Add mouse_name as hover text
                hovertemplate="<b>%{text}</b><br>"  # Bold mouse name
                + "Value: %{y:.1%}<br>"  # Format as percentage
                + "Area: %{customdata}<br>"  # Show area name
                + "<extra></extra>",  # Remove trace name from hover
                customdata=group_data["area_name"],  # Area name for hover
                showlegend=True,
            )
        )

    # Update layout
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="", tickangle=45, tickfont=dict(size=14)),
        yaxis=dict(
            title=dict(text=y_axis_title, font=dict(size=16)),
            tickfont=dict(size=14),
            tickformat=".0%",
        ),
        legend=dict(font=dict(size=14)),
        width=1200,
        height=700,
    )

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subregion fluorescence analysis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv", nargs="+", help="CSV file(s) to load")
    group.add_argument("--csv-dir", help="Directory containing CSV files")
    parser.add_argument(
        "--method", choices=["anova", "ttest"], default="anova", help="Statistical test"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output figure filename"
    )
    parser.add_argument("--static", action="store_true", help="Run unit tests")
    args = parser.parse_args()

    # Determine the file list
    if args.csv:
        files = args.csv
    elif args.csv_dir:
        files = glob.glob(os.path.join(args.csv_dir, "*.csv"))
    else:
        files = [DEFAULT_CSV]
    # load and concatenate
    dfs = [load_and_prepare(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    print(df_all.to_string())
    # compute and plot
    # stats_res = compute_statistics(df_all, method=args.method)
    if args.static:
        plot_results(df_all, args.output)
    else:
        plot_results_interactive(df_all, args.output)
