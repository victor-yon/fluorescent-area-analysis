import os
import sys
import glob
import argparse
from typing import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import stats
import tempfile
import unittest
import seaborn as sns

# --- Default Configuration ---
DEFAULT_CSV = 'results.csv'
DEFAULT_OUTPUT = 'subregion_analysis.png'
GROUP_ORDER = ['naive control', 'rotarod control', 'rotarod']
BAR_COLORS = {
    'naive control': '#0d3b66',  # Dark Blue
    'rotarod control': '#17becf',  # Turquoise
    'rotarod': '#2ca02c'  # Green
}


def infer_group_from_mouse_name(mouse_name):
    """
    Determine group based on substrings in mouse_id.
    """
    lname = str(mouse_name).lower()
    if 'naive control' in lname:
        return 'naive control'
    elif 'rotarod control' in lname:
        return 'rotarod control'
    elif 'rotarod' in lname:
        return 'rotarod'
    else:
        return 'unknown'


def load_and_prepare(csv_path):
    """
    Load a single CSV and compute mean fluorescence per mouse and subregion.
    Returns DataFrame with columns [subregion, group, mouse_id, mean_fluorescence, slice_number].
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Derive group if missing
    if 'group' not in df.columns:
        df['group'] = df['mouse_name'].apply(infer_group_from_mouse_name)

    # Extract slice number from area_name if present
    df['slice_number'] = df['area_name'].str.extract(r'slice(\d+)', expand=False)
    df['slice_number'] = pd.to_numeric(df['slice_number'], errors='coerce')

    # Remove sliceX text from area_name
    df['area_name'] = (
        df['area_name']
        .str.replace(r'_slice\d+', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    # Aggregate the different slices together
    if 'roi_rate' in df.columns:
        df_agg = (
            df.groupby(['area_name', 'group', 'mouse_name'], as_index=False)
            .agg(mean_roi_rate=('roi_rate', 'mean'),
                 std_roi_rate=('roi_rate', 'std'),
                 n_slices=('slice_number', 'nunique'))
        )
    elif 'particles_rate' in df.columns:
        df_agg = (
            df.groupby(['area_name', 'group', 'mouse_name'], as_index=False)
            .agg(mean_particles_rate=('particles_rate', 'mean'),
                 std_particles_rate=('particles_rate', 'std'),
                 n_slices=('slice_number', 'nunique'))
        )
    else:
        raise ValueError("No valid columns found for aggregation.")

    return df_agg

def plot_results(df_agg, output_path):
    """
    Plot bars + scatter, filtered and ordered, with colors.
    Y-axis scaled to percent (0–100).
    """
    # filter subregions by keywords
    keywords = ['lobule']
    pat = '|'.join(keywords)
    df_plot = df_agg[df_agg['area_name'].str.contains(pat, case=False, na=False)].copy()
    if df_plot.empty:
        print("Warning: no matching subregions, empty plot.")

    if 'mean_roi_rate' in df_plot.columns and 'mean_particles_rate' in df_plot.columns:
        raise ValueError("Both 'roi_rate' and 'particles_rate' columns found. It should be only one.")

    if 'mean_roi_rate' in df_plot.columns:
        plot_type = 'roi_rate'
    elif 'mean_particles_rate' in df_plot.columns:
        plot_type = 'particles_rate'
    else:
        raise ValueError("No valid columns found for plotting.")

    plt.figure(figsize=(16, 9))
    g = sns.barplot(
        data=df_plot, x="area_name", y=f"mean_{plot_type}", hue="group", errorbar="se", palette="dark", alpha=.6
    )

    sns.stripplot(
        data=df_plot, x="area_name", y=f"mean_{plot_type}", hue="group",
        linewidth=0.5, edgecolor='black', dodge=True, alpha=0.6, ax=g
    )

    # Remove extra legend handles
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[:3], labels[:3], title='', fontsize=14)

    # Configure axes
    plt.xticks(fontsize=14)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
    g.set_xlabel('', fontsize=16)
    plt.yticks(fontsize=14)

    if plot_type == 'roi_rate':
        g.set_ylabel('Fluorescent area of ROI', fontsize=16)
        g.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        g.set_ylim(0, 1)
    else:
        g.set_ylabel('Ratio of active cells in ROI', fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subregion fluorescence analysis')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--csv', nargs='+', help='CSV file(s) to load')
    group.add_argument('--csv-dir', help='Directory containing CSV files')
    parser.add_argument('--method', choices=['anova', 'ttest'], default='anova',
                        help='Statistical test')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output figure filename')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0]])
    else:
        # determine file list
        if args.csv:
            files = args.csv
        elif args.csv_dir:
            files = glob.glob(os.path.join(args.csv_dir, '*.csv'))
        else:
            files = [DEFAULT_CSV]
        # load and concatenate
        dfs = [load_and_prepare(f) for f in files]
        df_all = pd.concat(dfs, ignore_index=True)
        # compute and plot
        # stats_res = compute_statistics(df_all, method=args.method)
        plot_results(df_all, args.output)

        print(df_all.to_string())