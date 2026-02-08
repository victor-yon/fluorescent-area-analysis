import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import PathPatch
from svgpath2mpl import parse_path


@dataclass(frozen=True)
class MapProperties:
    title: str
    svg_path: Path
    areas: dict[str, str] = field(default_factory=dict)


MAPS: dict[str, MapProperties] = {
    "GCL hemispheres": MapProperties(
        title="Granule Cell Layer Hemisphere",
        svg_path=Path("../../resources/hemisphere_outline_GCL.svg"),
        areas={
            "simplex-center": "Simplex Center",
            "crusI-center": "Crus I Center",
            "crusII-center": "Crus II Center",
            "paramedian-left": "Paramedian Left",
            "paramedian-right": "Paramedian Right",
        },
    ),
    "GCL vermis": MapProperties(
        title="Granule Cell Layer Vermis",
        svg_path=Path("../../resources/vermis_outline_GCL.svg"),
        areas={
            "lobule2-center": "Lobule 2 Center",
            "lobule3-center": "Lobule 3 Center",
            "lobule3-left": "Lobule 3 Left",
            "lobule3-right": "Lobule 3 Right",
            "lobule45-center": "Lobule 4/5 Center",
            "lobule45-left": "Lobule 4/5 Left",
            "lobule45-right": "Lobule 4/5 Right",
            "lobule6-center": "Lobule 6 Center",
            "lobule6-left": "Lobule 6 Left",
            "lobule6-right": "Lobule 6 Right",
            "lobule7-dorsal": "Lobule 7 Dorsal",
            "lobule7-ventral": "Lobule 7 Ventral",
            "lobule8-left": "Lobule 8 Left",
            "lobule8-right": "Lobule 8 Right",
            "lobule9-center": "Lobule 9 Center",
            "lobule9-left": "Lobule 9 Left",
            "lobule9-right": "Lobule 9 Right",
            "lobule10-center": "Lobule 10 Center",
        },
    ),
    "MLI hemispheres": MapProperties(
        title="Molecular Layer Hemisphere",
        svg_path=Path("../../resources/hemisphere_outline_MLI.svg"),
        areas={
            "simplex-center": "Simplex Center",
            "crusI-left": "Crus I Left",
            "crusI-right": "Crus I Right",
            "crusII-left": "Crus II Left",
            "crusII-right": "Crus II Right",
            "paramedian-left": "Paramedian Left",
            "paramedian-right": "Paramedian Right",
        },
    ),
    "MLI vermis": MapProperties(
        title="Molecular Layer Vermis",
        svg_path=Path("../../resources/vermis_outline_MLI.svg"),
        areas={
            "lobule2-center": "Lobule 2 Center",
            "lobule3-center": "Lobule 3 Center",
            "lobule3-left": "Lobule 3 Left",
            "lobule3-right": "Lobule 3 Right",
            "lobule45-center_left": "Lobule 4/5 Center Left",
            "lobule45-center_right": "Lobule 4/5 Center Right",
            "lobule45-left": "Lobule 4/5 Left",
            "lobule45-right": "Lobule 4/5 Right",
            "lobule6-center": "Lobule 6 Center",
            "lobule6-left": "Lobule 6 Left",
            "lobule6-right": "Lobule 6 Right",
            "lobule7-dorsal": "Lobule 7 Dorsal",
            "lobule7-ventral": "Lobule 7 Ventral",
            "lobule8-left": "Lobule 8 Left",
            "lobule8-right": "Lobule 8 Right",
            "lobule9-center_left": "Lobule 9 Center Left",
            "lobule9-center_right": "Lobule 9 Center Right",
            "lobule9-left": "Lobule 9 Left",
            "lobule9-right": "Lobule 9 Right",
            "lobule10-center": "Lobule 10 Center",
        },
    ),
}


def plot_color_map(
    experiment_data_csv: str | Path,
    map_name: Literal["GCL hemispheres", "GCL vermis", "MLI hemispheres", "MLI vermis"],
    group_name: Literal[
        "Rotarod", "Rotarod control", "Naive control", "Vehicle control"
    ],
    lower_bound: float = 0.0,
    upper_bound: float = 0.6,
    show_title: bool = False,
    show_colorbar: bool = False,
    output_directory: str | Path = "out",
) -> None:
    """
    Generate a cerebellar color map by coloring SVG regions based on fluorescence data.

    Reads a CSV file with per-area average values, filters by experimental group, maps values
    to the cividis colormap, and renders the corresponding SVG template with colored regions
    as a PNG image.

    :param experiment_data_csv: Path to a CSV file with columns: Area, Group, Average.
    :param map_name: Which cerebellar map template to use (must match a key in MAPS).
    :param group_name: Experimental group to filter from the CSV.
    :param lower_bound: Minimum value for colormap normalization.
    :param upper_bound: Maximum value for colormap normalization.
    :param show_title: Whether to display the map title above the figure.
    :param show_colorbar: Whether to display the colorbar legend.
    :param output_directory: Directory where the output PNG will be saved.
    """
    props = MAPS[map_name]
    svg_path = Path(__file__).parent / props.svg_path

    # Read CSV and filter by group
    df = pd.read_csv(experiment_data_csv)
    df_group = df[df["Group"].str.lower() == group_name.lower()]

    # Map area names to their average values
    area_averages = df_group.set_index("Area")["Average"]

    # Parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    inkscape_label = "{http://www.inkscape.org/namespaces/inkscape}label"
    svg_path_tag = "{http://www.w3.org/2000/svg}path"

    # Setup colormap
    cmap = plt.cm.cividis
    norm = plt.Normalize(vmin=lower_bound, vmax=upper_bound)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 14))

    # Draw all SVG paths
    for svg_elem in root.iter(svg_path_tag):
        label = svg_elem.get(inkscape_label)
        d = svg_elem.get("d")
        if not d:
            continue

        mpl_path = parse_path(d)
        # SVG y-axis is inverted relative to matplotlib
        mpl_path.vertices[:, 1] *= -1

        # Parse original SVG style properties
        style = svg_elem.get("style", "")
        style_dict = dict(
            item.split(":", 1) for item in style.split(";") if ":" in item
        )
        svg_stroke = style_dict.get("stroke", "none")
        svg_fill = style_dict.get("fill", "none")
        svg_lw = float(style_dict.get("stroke-width", "0.265")) * 2

        if label in props.areas and label in area_averages.index:
            # Colored fill for data areas
            value = area_averages[label]
            facecolor = cmap(norm(value))
            edgecolor = svg_stroke if svg_stroke != "none" else "black"
            patch = PathPatch(
                mpl_path, facecolor=facecolor, edgecolor=edgecolor, lw=svg_lw
            )
        else:
            # Preserve original SVG colors for background paths
            facecolor = svg_fill if svg_fill != "none" else "none"
            edgecolor = svg_stroke if svg_stroke != "none" else "none"
            patch = PathPatch(
                mpl_path, facecolor=facecolor, edgecolor=edgecolor, lw=svg_lw
            )
        ax.add_patch(patch)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if show_title:
        ax.set_title(f"{props.title} — {group_name}", fontsize=28)

    # Add colorbar
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.04, shrink=0.7)
        cbar.set_label("Average ROI rate", fontsize=22)
        cbar.ax.tick_params(labelsize=18)

    # Save output
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_directory
        / f"{Path(experiment_data_csv).stem}_{group_name}_{map_name}.png"
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Hemisphere map saved to: {output_path.resolve().absolute()}")


if __name__ == "__main__":
    # Example usage:
    plot_color_map(
        experiment_data_csv="../../data/Experiment1_particles_with_colors_MLI.csv",
        map_name="MLI hemispheres",
        group_name="Rotarod",
        lower_bound=0.0,
        upper_bound=0.6,
        show_title=True,
        show_colorbar=True,
        output_directory="../../out",
    )
