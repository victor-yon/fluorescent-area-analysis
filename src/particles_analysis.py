from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy._typing import NDArray
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
)
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import regionprops, perimeter_crofton

# =============================================================================
# ANALYSIS OVERVIEW
# =============================================================================
# This script counts cells and determines what fraction express an
# immediate early gene (IEG) marker within a defined brain region (ROI).
#
# For each image pair (DAPI + IEG channel), the pipeline is:
#
#   1. DAPI channel  → identify and count all cell nuclei
#                       (blur to reduce noise → threshold to separate nuclei
#                        from background → watershed to split touching nuclei
#                        → filter by shape to remove non-cell artifacts)
#
#   2. IEG channel   → for each DAPI-identified nucleus, check whether
#                       enough of its area has IEG signal above threshold
#                       (cells meeting this criterion are counted as IEG+)
#
#   3. Output        → nb_particles_dapi    : total nuclei counted
#                      nb_particles_ieg_overlapping_surface : IEG+ nuclei
#                      particles_rate       : IEG+ / total  (used for plots)
#                      nb_particles_ieg_threshold_watershed : alternative
#                        IEG count for reference (independent blob detection,
#                        not anchored to DAPI nuclei)
# =============================================================================

from commun import batch_iterator, get_roi_mask, get_threshold_mask

# Imports for plotting the image files
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage.segmentation import find_boundaries
from skimage import exposure

def _filter_labels_by_circularity(
    labels: NDArray,
    min_circularity: float,
    max_circularity: float,
    use_crofton: bool = True,
) -> NDArray:
    """
    Keep only labeled objects whose circularity falls in [min_circularity, max_circularity].
    Circularity = 4*pi*Area / Perimeter^2 (ImageJ definition).

    We use Crofton perimeter by default for better stability on pixel grids.
    """
    # Circularity ranges from 0 (elongated/irregular) to 1 (perfect circle).
    # Cell nuclei are roughly circular, so this filter removes debris, clumped
    # cells, or other artifacts that have an unusual shape.
    # A typical setting of 0.70–1.00 keeps round nuclei and rejects elongated objects.
    if min_circularity <= 0 and max_circularity >= 1:
        return labels  # circularity filter is disabled — keep everything

    keep_ids = []
    for r in regionprops(labels):
        area = r.area
        if area == 0:
            continue
        if use_crofton:
            # 4-direction Crofton perimeter is a good, scale-stable choice
            perim = perimeter_crofton(r.image, directions=4)
        else:
            # fallback to pixel-trace perimeter if desired
            perim = r.perimeter

        if perim <= 0:
            circ = 0.0
        else:
            circ = 4.0 * np.pi * area / (perim ** 2)

        if (circ >= min_circularity) and (circ <= max_circularity):
            keep_ids.append(r.label)

    if not keep_ids:
        return labels * 0  # all removed

    mask_keep = np.isin(labels, keep_ids)
    return labels * mask_keep



def particles_batch_processing(
    data_dir: Path | str,
    mouse_filter: str = "*",
    area_filter: str = "*",
    dapi_threshold: int = 1624,
    ieg_threshold: int = 800,
    gaussian_sigma: float = 2,
    min_particle_size: float = 30,
    markers_percentile: float = 90,
    rolling_ball_radius: float | None = 90,
    use_cache: bool = True,
    min_circularity: float = 0.0,   # NEW (ImageJ-style range 0..1)
    max_circularity: float = 1.0,   # NEW
) -> Dict[str, List[Any]]:
    """
    Analyze all the data from the directories matching the filters.
    The folder structure is assumed to be : data_dir/mouse_dir/area_dir/Default
    With inside:
        - exactly one data file matching the channel number as "img_channel00N*"
        - exactly one ROI file matching "*.roi"

    :param data_dir: The relative or absolute path to the directory that contains all the data.
    :param mouse_filter: The pattern to select the mouse subfolder to process. Where '*' is a wildcard.
    :param area_filter: The pattern to select the area subfolder to process. Where '*' is a wildcard.
    :param dapi_threshold: The detection threshold value for the DAPI channel pixels.
    :param ieg_threshold: The detection threshold value for the IEG channel pixels.
    :param gaussian_sigma: The sigma value for the Gaussian filter applied to the data before processing.
    :param min_particle_size: The minimum size of particles to be considered valid.
    :param markers_percentile: The percentile value used to generate markers for the watershed segmentation.
    :param rolling_ball_radius: The radius for the rolling ball background subtraction for pre-processing.
        If None, no background subtraction is applied.
    :param use_cache: If True, use cached data if available to speed up processing.

    :return: The statistics as a dictionary where the keys are the field names and the values are lists of values.
    """
    # These lists will be filled with one value per image/ROI processed,
    # then written out as columns in the results CSV.
    results: Dict[str, List[Any]] = {
        "mouse_name": [],
        "area_name": [],
        "roi_area_px": [],
        "nb_particles_dapi": [],
        "nb_particles_ieg_overlapping_surface": [],
        "nb_particles_ieg_threshold_watershed": [],
        "particles_rate": [],
    }

    # Iterate over every (mouse, brain region) image pair found in data_dir
    for roi, img_data, area_name, mouse_name in batch_iterator(
        data_dir,
        mouse_filter,
        area_filter,
        ieg_channel=True,
        dapi_channel=True,
        rolling_ball_radius=rolling_ball_radius,
        use_cache=use_cache,
    ):
        # Record the size of the ROI in pixels (useful for normalisation checks)
        roi_mask = get_roi_mask(img_data["dapi"], roi)
        results["roi_area_px"].append(int(np.count_nonzero(roi_mask)))

        # --- STEP 1: Count all nuclei in the DAPI channel ---
        # This returns the total cell count AND a labelled map where every
        # detected nucleus has a unique ID number (used in step 2).
        num_particles_dapi, labels = processing_threshold_watershed(
            img_data["dapi"],
            roi,
            threshold=dapi_threshold,
            gaussian_sigma=gaussian_sigma,
            min_particle_size=min_particle_size,
            markers_percentile=markers_percentile,
            silent=True,
            min_circularity=min_circularity,
            max_circularity=max_circularity,
        )
        results["nb_particles_dapi"].append(num_particles_dapi)

        # --- STEP 2a: Count IEG+ nuclei (primary method, used for particles_rate) ---
        # For each DAPI nucleus identified above, check whether its pixels in
        # the IEG channel have sufficient signal above threshold.
        # A nucleus is counted as IEG+ if the fraction of its area above
        # threshold exceeds min_overlap_ratio (default 10%).
        results["nb_particles_ieg_overlapping_surface"].append(
            processing_threshold_overlapping_surface(
                img_data["ieg"],
                gaussian_sigma=gaussian_sigma,
                labels=labels,
                threshold=ieg_threshold,
                silent=True,
            )
        )

        # --- STEP 2b: Count IEG blobs independently (reference method only) ---
        # Detects IEG-positive regions by the same blur → threshold → watershed
        # pipeline used for DAPI, but applied to the IEG channel across the
        # whole ROI without reference to individual DAPI nuclei.
        # Kept for comparison; particles_rate is derived from step 2a above.
        results["nb_particles_ieg_threshold_watershed"].append(
            processing_threshold_watershed(
                img_data["ieg"],
                roi,
                threshold=ieg_threshold,
                gaussian_sigma=gaussian_sigma,
                min_particle_size=min_particle_size,
                markers_percentile=markers_percentile,
                silent=True,
                min_circularity=min_circularity,
                max_circularity=max_circularity,
            )[0]
        )

        # --- STEP 3: Compute the IEG expression rate for this image ---
        results["mouse_name"].append(mouse_name)
        results["area_name"].append(area_name)

        if results["nb_particles_ieg_overlapping_surface"][-1] == 0:
            warning(
                f"No particles found in IEG channel {mouse_name} - {area_name} with threshold {ieg_threshold}."
            )
        if results["nb_particles_dapi"][-1] == 0:
            warning(
                f"No particles found in DAPI channel {mouse_name} - {area_name} with threshold {dapi_threshold}."
            )
            results["particles_rate"].append(0)
        else:
            # particles_rate = fraction of DAPI nuclei that are IEG positive
            results["particles_rate"].append(
                results["nb_particles_ieg_overlapping_surface"][-1] / results["nb_particles_dapi"][-1]
            )

    return results


def _threshold_watershed(
    data: NDArray,
    min_particle_size: float,
    markers_percentile: float,
    silent: bool = False,
    min_circularity: float = 0.0,
    max_circularity: float = 1.0,
) -> Tuple[int, NDArray]:
    """
    Internal function to apply watershed segmentation on the masked data.

    :param masked_data: The binary data to process.
    :param min_particle_size: The minimum size of particles to be considered valid.
    :param markers_percentile: The percentile value used to generate markers for the watershed
        segmentation.
    :param silent: If True, suppress the output messages.

    :return: The number of particles detected and the labels of the segmented particles.
    """
    # For each foreground pixel, compute its distance to the nearest background
    # pixel. Pixel values are high at the centre of a nucleus and low at the edges.
    distance = distance_transform_edt(data)

    # Identify seed points (markers) at the brightest peaks of the distance map.
    # Using the Nth percentile avoids placing too many seeds in flat regions,
    # which would over-segment nuclei.
    markers = label(distance > np.percentile(distance, markers_percentile))

    # Watershed: starting from each seed, "flood" outward until regions meet.
    # This separates touching or overlapping nuclei into individual objects.
    labels_ws = watershed(-distance, markers, mask=data)

    # Discard objects that are too small to be real nuclei (likely debris or noise)
    labels_ws = remove_small_objects(labels_ws, min_size=min_particle_size)

    # Discard objects whose shape is too irregular to be a nucleus
    labels_ws = _filter_labels_by_circularity(
        labels_ws, min_circularity=min_circularity, max_circularity=max_circularity
    )

    # Each remaining object has a unique integer label; background is 0
    num_particles = len(np.unique(labels_ws)) - 1  # subtract 1 to exclude background

    if not silent:
        print(f"Number of particles: {num_particles:,d}")

    return num_particles, labels_ws


def processing_threshold_watershed(
    data: NDArray,
    roi: NDArray,
    threshold: int,
    gaussian_sigma: float,
    min_particle_size: float,
    markers_percentile: float,
    silent: bool = False,
    min_circularity: float = 0.0,   # NEW
    max_circularity: float = 1.0,   # NEW
) -> Tuple[int, NDArray]:
    """
    Process the data to count the number of particles in a given ROI using watershed segmentation.
    Also returns the labels of the segmented particles for further analysis.

    :param data: The data to process, typically an image array.
    :param roi: The region of interest (ROI) mask to apply on the data.
    :param threshold: The detection threshold value for the pixels.
    :param gaussian_sigma: The sigma value for the Gaussian filter applied to the data before processing.
    :param min_particle_size: The minimum size of particles to be considered valid.
    :param markers_percentile: The percentile value used to generate markers for the watershed segmentation.
    :param silent: If True, suppress the output messages.

    :return: The number of particles detected in the ROI and the labels of the segmented particles.
    """

    # Blur the image slightly to reduce pixel-level noise before thresholding
    blurred = gaussian_filter(data, sigma=gaussian_sigma)

    # Create a binary mask: True where pixel intensity is above threshold
    thr_mask = get_threshold_mask(blurred, threshold)
    # Create a binary mask: True where pixel is inside the ROI polygon
    roi_mask = get_roi_mask(blurred, roi)
    # Keep only pixels that are both above threshold AND inside the ROI
    thr_and_roi_mask = np.logical_and(thr_mask, roi_mask)

    return _threshold_watershed(
        thr_and_roi_mask,
        min_particle_size,
        markers_percentile,
        silent,
        min_circularity=min_circularity,
        max_circularity=max_circularity,
    )


def processing_threshold_overlapping_surface(
    data: NDArray,
    gaussian_sigma: float,
    labels: NDArray,
    threshold: int,
    min_overlap_ratio: float | None = 0.1,
    silent: bool = False,
) -> int:
    """
    Process the data to count the number of particles with intensity above a threshold
    that are overlapping with a previous watershed segmentation.

    :param data: The data to process, typically an image array.
    :param gaussian_sigma: The sigma value for the Gaussian filter applied to the data before
        processing.
    :param labels: The labels of the particles from previous segmentation (e.g., DAPI channel).
    :param threshold: The detection threshold value for the pixels.
    :param min_overlap_ratio: The minimum ratio of pixels above the intensity threshold
        within a particle to consider it as positive. If None, any overlap counts.
    :param silent: If True, suppress the output messages.
    :return: The number of particles.
    """
    # Blur and threshold the IEG channel to find pixels with signal above background
    blurred = gaussian_filter(data, sigma=gaussian_sigma)
    thr_mask = get_threshold_mask(blurred, threshold)

    # Get the ID of each DAPI nucleus (the labelled map from step 1)
    unique_labels = np.unique(labels[labels != 0])

    # For each nucleus, decide whether it is IEG positive
    num_particles_with_ieg = 0
    for label_id in unique_labels:
        particle_mask = labels == label_id  # pixels belonging to this nucleus

        if min_overlap_ratio is None:
            # IEG+  if ANY pixel in the nucleus is above threshold
            if np.any(thr_mask[particle_mask]):
                num_particles_with_ieg += 1
        else:
            # IEG+  if the fraction of pixels above threshold exceeds min_overlap_ratio
            # e.g. default 0.10 means at least 10% of the nucleus must be IEG bright
            if np.sum(thr_mask[particle_mask]) / np.sum(particle_mask) > min_overlap_ratio:
                num_particles_with_ieg += 1

    if not silent:
        print(f"Number of particles: {num_particles_with_ieg:,d}")

    return num_particles_with_ieg


# Setup to plot images
def _axis_gray(ax, img, title=None, clip_percentiles=(2,98), gamma=1.0, gain=1.0):
    disp = to_display_gray(img, clip_percentiles=clip_percentiles, gamma=gamma, gain=gain)
    ax.imshow(disp, cmap="gray")
    if title: ax.set_title(title, fontsize=10)
    ax.axis("off")

def _plot_yellow_boundaries(ax, labels_or_mask, linewidth=1.8):
    # Accepts a labels array or a boolean mask; computes boundaries then draws yellow contours
    if labels_or_mask.dtype == bool:
        bnd = labels_or_mask
    else:
        from skimage.segmentation import find_boundaries
        bnd = find_boundaries(labels_or_mask, mode="inner")
    # Use a contour to get crisp lines:
    ax.contour(bnd.astype(float), levels=[0.5], colors="yellow", linewidths=linewidth)

def _axis_labels(ax, gray_img, labels, title=None, clip_percentiles=(2,98), gamma=1.0, gain=1.0):
    disp = to_display_gray(gray_img, clip_percentiles=clip_percentiles, gamma=gamma, gain=gain)
    ax.imshow(disp, cmap="gray")
    _plot_yellow_boundaries(ax, labels)  # yellow outlines only
    if title: ax.set_title(title, fontsize=10)
    ax.axis("off")


# New function to plot images
def save_all_debug_grids(
    data_dir: Path | str,
    out_dir: Path | str,
    dapi_threshold: int,
    ieg_threshold: int,
    gaussian_sigma: float,
    min_particle_size: float,
    markers_percentile: float,
    rolling_ball_radius: float | None = 90,
    mouse_filter: str = "*",
    area_filter: str = "*",
    min_circularity: float = 0.0,
    max_circularity: float = 1.0,
    use_cache: bool = True,
clip_percentiles=(2, 98),
    gamma: float = 1.0,
    gain: float = 1.2,
) -> None:
    """
    Save a single 7-panel PNG per (mouse, area):
      (1) Raw DAPI
      (2) Raw IEG
      (3) DAPI preproc (blur + thr + watershed boundaries)
      (4) IEG  preproc (blur + thr + watershed boundaries)
      (5) DAPI particles (size + circularity + ROI)
      (6) IEG particles (threshold + watershed) -> nb_particles_ieg_threshold_watershed
      (7) IEG particles overlapping DAPI (overlapping surface) -> nb_particles_ieg_overlapping_surface
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for roi, img_data, area_name, mouse_name in batch_iterator(
        data_dir,
        mouse_filter,
        area_filter,
        ieg_channel=True,
        dapi_channel=True,
        rolling_ball_radius=rolling_ball_radius,
        use_cache=use_cache,
    ):
        dapi = img_data["dapi"]
        ieg  = img_data["ieg"]

        # Preproc: blur + thr + roi
        dapi_blur = gaussian_filter(dapi, sigma=gaussian_sigma)
        dapi_thr  = get_threshold_mask(dapi_blur, dapi_threshold)
        dapi_roi  = get_roi_mask(dapi_blur, roi)
        dapi_mask = np.logical_and(dapi_thr, dapi_roi)

        ieg_blur = gaussian_filter(ieg, sigma=gaussian_sigma)
        ieg_thr  = get_threshold_mask(ieg_blur, ieg_threshold)
        ieg_roi  = get_roi_mask(ieg_blur, roi)
        ieg_mask = np.logical_and(ieg_thr, ieg_roi)

        # Watershed DAPI (size + circularity)
        dapi_num, dapi_labels = _threshold_watershed(
            dapi_mask,
            min_particle_size=min_particle_size,
            markers_percentile=markers_percentile,
            silent=True,
            min_circularity=min_circularity,
            max_circularity=max_circularity,
        )

        # IEG (threshold + watershed)  -> csv: nb_particles_ieg_threshold_watershed
        ieg_thrws_num, ieg_thrws_labels = _threshold_watershed(
            ieg_mask,
            min_particle_size=min_particle_size,
            markers_percentile=markers_percentile,
            silent=True,
            min_circularity=min_circularity,
            max_circularity=max_circularity,
        )

        # IEG overlapping DAPI (overlapping surface) -> csv: nb_particles_ieg_overlapping_surface
        particle_mask = dapi_labels > 0
        thr_and_particle_mask = np.logical_and(ieg_thr, particle_mask)
        ieg_ov_num, ieg_ov_labels = _threshold_watershed(
            thr_and_particle_mask,
            min_particle_size=min_particle_size,
            markers_percentile=markers_percentile,
            silent=True,
            min_circularity=min_circularity,
            max_circularity=max_circularity,
        )

        # Figure layout: 3 rows x 3 cols (last two cells used for a small legend/notes)
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.02, hspace=0.10)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        # 1) Raw DAPI
        _axis_gray(ax1, dapi, "1) Raw DAPI", clip_percentiles, gamma, gain)
        # 2) Raw IEG
        _axis_gray(ax2, ieg, "2) Raw IEG", clip_percentiles, gamma, gain)
        # 3) DAPI preproc
        _axis_labels(ax3, dapi_blur, dapi_labels, f"3) DAPI preproc (n={dapi_num})", clip_percentiles, gamma, gain)
        # 4) IEG preproc
        _axis_labels(ax4, ieg_blur, ieg_thrws_labels, f"4) IEG preproc (n={ieg_thrws_num})", clip_percentiles, gamma,
                      gain)
        # 5) DAPI particles
        _axis_labels(ax5, dapi_blur, dapi_labels, f"5) DAPI particles (n={dapi_num})", clip_percentiles, gamma, gain)
        # 6) IEG thr+watershed
        _axis_labels(ax6, ieg_blur, ieg_thrws_labels, f"6) IEG thr+watershed (n={ieg_thrws_num})", clip_percentiles,
                     gamma, gain)
        # 7) IEG overlapping DAPI
        _axis_labels(ax7, ieg_blur, ieg_ov_labels, f"7) IEG overlapping DAPI (n={ieg_ov_num})", clip_percentiles, gamma,
                     gain)

        # 8–9) Notes panel (parameters)
        ax8.axis("off")
        ax9.axis("off")
        txt = (
            f"Mouse: {mouse_name}\nArea: {area_name}\n\n"
            f"DAPI thr: {dapi_threshold} | IEG thr: {ieg_threshold}\n"
            f"Gaussian σ: {gaussian_sigma}\n"
            f"Min size: {min_particle_size} px\n"
            f"Markers pct: {markers_percentile}\n"
            f"Circularity: [{min_circularity:.2f}, {max_circularity:.2f}]\n"
            f"Rolling-ball radius: {rolling_ball_radius}"
        )
        ax8.text(0.0, 0.5, txt, va="center", ha="left", fontsize=10, family="monospace")
        fig.suptitle(f"{mouse_name} • {area_name}", y=0.995, fontsize=12)

        # Save as a single PNG per datapoint
        sample_dir = out_dir / f"{mouse_name}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"{area_name}_grid.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

def to_display_gray(img, clip_percentiles=(2, 98), gamma=1.0, gain=1.0):
    """
    Prepare an image for display:
      - Contrast stretch to [0,1] using robust percentiles
      - Optional gamma correction
      - Optional gain to brighten after gamma
    Returns float image in [0,1].
    """
    p_low, p_high = np.percentile(img, clip_percentiles)
    disp = exposure.rescale_intensity(img, in_range=(p_low, p_high), out_range=(0, 1)).astype(np.float32)
    if gamma != 1.0:
        disp = exposure.adjust_gamma(disp, gamma=gamma)
    if gain != 1.0:
        disp = np.clip(disp * gain, 0, 1)
    return disp
