from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy._typing import NDArray
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    gaussian_laplace,
    grey_dilation,
)
from skimage.measure import label
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed

from src.commun import batch_iterator, get_roi_mask, get_threshold_mask
from src.plots import plot_data


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
    # Prepare the result dictionary with empty lists
    results: Dict[str, List[Any]] = {
        "mouse_name": [],
        "area_name": [],
        "nb_particles_dapi": [],
        "nb_particles_ieg": [],
        "nb_particles_ieg_gaussian_laplace": [],
        "nb_particles_ieg_threshold": [],
        "particles_rate": [],
    }

    # Files iterator
    for roi, img_data, area_name, mouse_name in batch_iterator(
        data_dir,
        mouse_filter,
        area_filter,
        ieg_channel=True,
        dapi_channel=True,
        rolling_ball_radius=rolling_ball_radius,
        use_cache=use_cache,
    ):
        # First channel (dapi)
        num_particles_dapi, labels = particles_processing_threshold(
            img_data["dapi"],
            roi,
            threshold=dapi_threshold,
            gaussian_sigma=gaussian_sigma,
            min_particle_size=min_particle_size,
            markers_percentile=markers_percentile,
            show_plot=False,
            silent=True,
        )
        results["nb_particles_dapi"].append(num_particles_dapi)

        # Second channel (ieg) with 3 methods
        results["nb_particles_ieg"].append(
            particles_processing_overlapping(
                img_data["ieg"],
                gaussian_sigma=gaussian_sigma,
                labels=labels,
                threshold=ieg_threshold,
                show_plot=False,
                silent=True,
            )
        )

        results["nb_particles_ieg_threshold"].append(
            particles_processing_threshold(
                img_data["ieg"],
                roi,
                threshold=ieg_threshold,
                gaussian_sigma=gaussian_sigma,
                min_particle_size=min_particle_size,
                markers_percentile=markers_percentile,
                show_plot=False,
                silent=True,
            )[0]
        )

        results["nb_particles_ieg_gaussian_laplace"].append(
            particles_processing_gaussian_laplace(
                img_data["ieg"],
                roi,
                gaussian_sigma=gaussian_sigma,
                labels=labels,
                show_plot=False,
                silent=True,
            )
        )

        # Compute statistics
        results["mouse_name"].append(mouse_name)
        results["area_name"].append(area_name)

        if results["nb_particles_ieg"][-1] == 0:
            warning(
                f"No particles found in IEG channel {mouse_name} - {area_name} with threshold {ieg_threshold}."
            )
        if results["nb_particles_dapi"][-1] == 0:
            warning(
                f"No particles found in DAPI channel {mouse_name} - {area_name} with threshold {dapi_threshold}."
            )
            results["particles_rate"].append(0)
        else:
            results["particles_rate"].append(
                results["nb_particles_ieg"][-1] / results["nb_particles_dapi"][-1]
            )

    return results


def particles_processing_threshold(
    data: NDArray,
    roi: NDArray,
    threshold: int,
    gaussian_sigma: float,
    min_particle_size: float,
    markers_percentile: float,
    show_plot: bool = True,
    save_plot_path: Path | str | None = None,
    silent: bool = False,
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
    :param show_plot: If True, display the plot of the data with the segmentation results.
    :param save_plot_path: If provided, save the plot to this path.
    :param silent: If True, suppress the output messages.

    :return: The number of particles detected in the ROI and the labels of the segmented particles.
    """

    # Apply Gaussian filter
    blurred = gaussian_filter(data, sigma=gaussian_sigma)

    # Apply thresholding and ROI mask
    thr_mask = get_threshold_mask(blurred, threshold)
    roi_mask = get_roi_mask(blurred, roi)
    thr_and_roi_mask = np.logical_and(thr_mask, roi_mask)

    # Compute the distance transform
    distance = distance_transform_edt(thr_and_roi_mask)

    # Generate markers, with the Nth percentile for strong peaks
    markers = label(distance > np.percentile(distance, markers_percentile))

    # Apply watershed segmentation
    labels = watershed(-distance, markers, mask=thr_and_roi_mask)
    labels = remove_small_objects(labels, min_size=min_particle_size)

    # Count the number of particles
    num_particles = len(np.unique(labels)) - 1

    if not silent:
        print(f"Number of particles: {num_particles:,d}")

    # Plot the data
    if show_plot or save_plot_path is not None:
        title = (
            f"Particles scan\nThreshold: {threshold}, Gaussian sigma: {gaussian_sigma}, "
            f"Min particle size: {min_particle_size}, Markers percentile: {markers_percentile}"
        )
        plot_data(
            data,
            roi,
            thr_mask,
            thr_and_roi_mask,
            labels,
            title=title,
            show_plot=show_plot,
            save_path=save_plot_path,
        )

    return num_particles, labels


def particles_processing_overlapping(
    data: NDArray,
    gaussian_sigma: float,
    labels: NDArray,
    threshold: int,
    min_overlap_ratio: float | None = None,
    show_plot: bool = True,
    save_plot_path: Path | str | None = None,
    silent: bool = False,
) -> int:
    # Apply Gaussian filter
    blurred = gaussian_filter(data, sigma=gaussian_sigma)

    # Apply thresholding and ROI mask
    thr_mask = get_threshold_mask(blurred, threshold)

    # Get unique particle labels (excluding background 0)
    unique_labels = np.unique(labels[labels != 0])

    num_particles_with_ieg = 0
    for label_id in unique_labels:
        # Create a mask for the current particle
        particle_mask = labels == label_id

        if min_overlap_ratio is None:
            # Check if any pixel in the particle is above the intensity threshold
            if np.any(thr_mask[particle_mask]):
                num_particles_with_ieg += 1
        # Check the ratio of pixels above the intensity threshold
        elif (
            np.sum(thr_mask[particle_mask]) / np.sum(particle_mask) > min_overlap_ratio
        ):
            num_particles_with_ieg += 1

    if not silent:
        print(f"Number of particles: {num_particles_with_ieg:,d}")

    return num_particles_with_ieg


def particles_processing_gaussian_laplace(
    data: NDArray,
    roi: NDArray,
    gaussian_sigma: float,
    labels: NDArray,
    expansion_radius: int = 5,
    intensity_threshold_sigma: float = 4.0,
    show_plot: bool = True,
    save_plot_path: Path | str | None = None,
    silent: bool = False,
) -> int:
    """
    Process the data to count the number of particles with IEG expression using robust detection.

    For each DAPI-labeled particle:
    1. Expand the particle area by a fixed radius (excluding other particles)
    2. Apply scale-normalized Laplacian-of-Gaussian to detect blob-like structures
    3. Count particles where LoG response is significantly above the background level

    :param data: The data to process, typically an image array.
    :param roi: The region of interest (ROI) coordinates.
    :param gaussian_sigma: The sigma value for the Gaussian filter applied to the data before processing.
    :param labels: The labels of the particles from DAPI channel processing.
    :param expansion_radius: Number of pixels to expand each particle area for detection (default: 5).
    :param intensity_threshold_sigma: Number of standard deviations above background mean to consider
        as significant signal (default: 4.0). Higher values are more conservative.
        This parameter should be tuned based on your specific imaging conditions and expected
        IEG expression levels.
    :param show_plot: If True, display the plot of the data with the segmentation results.
    :param save_plot_path: If provided, save the plot to this path.
    :param silent: If True, suppress the output messages.

    :return: The number of particles with IEG expression.
    """
    # Convert to float to avoid overflow errors with integer types
    data_float = data.astype(np.float64)

    # Apply Gaussian filter to smooth noise
    data_filtered = gaussian_filter(data_float, sigma=gaussian_sigma)

    # Apply scale-normalized Laplacian-of-Gaussian
    # The normalization (sigma^2 * LoG) makes the response scale-invariant
    log_response = -(gaussian_sigma**2) * gaussian_laplace(
        data_filtered, sigma=gaussian_sigma
    )

    # Get ROI mask to constrain analysis
    roi_mask = get_roi_mask(data, roi)

    # Get unique particle labels (excluding background 0)
    unique_labels = np.unique(labels[labels != 0])

    # Compute background statistics (pixels in ROI but outside any particle)
    background_mask = roi_mask & (labels == 0)
    log_background = log_response[background_mask]
    bg_mean = np.mean(log_background)
    bg_std = np.std(log_background)

    # Threshold is background mean + N standard deviations
    log_threshold = bg_mean + intensity_threshold_sigma * bg_std

    # Dilate all particle labels at once using grey dilation (preserves label values)
    # This is much faster than dilating each particle individually
    selem = disk(expansion_radius)
    dilated_labels = grey_dilation(labels, footprint=selem)

    num_particles_with_ieg = 0
    for label_id in unique_labels:
        # Get the expanded mask for this particle from the dilated labels
        # This includes all pixels that were dilated into from this particle
        expanded_mask = dilated_labels == label_id

        # Also include the original particle pixels
        original_mask = labels == label_id
        expanded_mask = expanded_mask | original_mask

        # Extract LoG response in the expanded area
        log_in_area = log_response[expanded_mask]

        # Check if there's a significant LoG response (indicating a blob/particle)
        # Use max LoG value in the expanded area
        if len(log_in_area) > 0 and np.max(log_in_area) > log_threshold:
            num_particles_with_ieg += 1

    if not silent:
        print(f"Number of particles: {num_particles_with_ieg:,d}")

    return num_particles_with_ieg


def save_all_particle_scans(
    data_dir: Path | str,
    out_dir: Path | str,
    threshold: int,
    gaussian_sigma: float,
    min_particle_size: float,
    markers_percentile: float,
    rolling_ball_radius: float | None = 90,
    mouse_filter: str = "*",
    area_filter: str = "*",
    use_cache: bool = True,
) -> None:
    """
    Process all the data from the directories matching the filters and save the particle scans.

    :param data_dir: The relative or absolute path to the directory that contains all the data.
    :param out_dir: The directory where the particle scan images will be saved.
    :param threshold: The detection threshold value for the pixels.
    :param gaussian_sigma: The sigma value for the Gaussian filter applied to the data before processing.
    :param min_particle_size: The minimum size of particles to be considered valid.
    :param markers_percentile: The percentile value used to generate markers for the watershed segmentation.
    :param rolling_ball_radius: The radius for the rolling ball background subtraction for pre-processing.
    :param mouse_filter: The pattern to select the mouse subfolder to process. Where '*' is a wildcard.
    :param area_filter: The pattern to select the area subfolder to process. Where '*' is a wildcard.
    :param use_cache: If True, use cached data if available to speed up processing.
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    for roi, img_data, area_name, mouse_name in batch_iterator(
        data_dir,
        mouse_filter,
        area_filter,
        ieg_channel=True,
        dapi_channel=True,
        rolling_ball_radius=rolling_ball_radius,
        use_cache=use_cache,
    ):
        img = img_data["ieg"]
        file_name = Path(f"particles_{data_dir.name}_{mouse_name}_{area_name}.png")
        particles_processing_threshold(
            img,
            roi,
            threshold,
            gaussian_sigma,
            min_particle_size,
            markers_percentile,
            show_plot=False,
            silent=True,
            save_plot_path=out_dir / file_name,
        )
