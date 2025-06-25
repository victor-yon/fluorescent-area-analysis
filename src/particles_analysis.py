from logging import warning
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from numpy._typing import NDArray
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from scipy.ndimage import gaussian_filter, distance_transform_edt

from commun import open_image, open_roi, get_threshold_mask, get_roi_mask, save_results, batch_iterator
from plots import plot_data


def particles_batch_processing(
        data_dir: Path | str,
        mouse_filter: str = '*',
        area_filter: str = '*',
        dapi_threshold: int= 1624,
        ieg_threshold: int = 800,
        gaussian_sigma: float = 2,
        min_particle_size: float = 30,
        markers_percentile: float = 90,
        rolling_ball_radius: float | None = 90,
        use_cache: bool = True
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
    results = {'mouse_name': [], 'area_name': [], 'nb_particles_dapi': [], 'nb_particles_ieg': [],
               'particles_rate': []}

    # Files iterator
    for roi, img_data, area_name, mouse_name in (
            batch_iterator(data_dir, mouse_filter, area_filter, ieg_channel=True, dapi_channel=True,
                           rolling_ball_radius=rolling_ball_radius, use_cache=use_cache)
    ):
        # Channel iterator for one file
        for channel_name, channel_data in img_data.items():
            results[f'nb_particles_{channel_name}'].append(particles_processing(
                channel_data,
                roi,
                threshold=ieg_threshold if channel_name == 'ieg' else dapi_threshold,
                gaussian_sigma=gaussian_sigma,
                min_particle_size=min_particle_size,
                markers_percentile=markers_percentile,
                show_plot=False,
                silent=True
            ))

        # Compute statistics
        results['mouse_name'].append(mouse_name)
        results['area_name'].append(area_name)

        if results[f'nb_particles_ieg'][-1] == 0:
            warning(f'No particles found in IEG channel {mouse_name} - {area_name} with threshold {ieg_threshold}.')
        if results[f'nb_particles_dapi'][-1] == 0:
            warning(f'No particles found in DAPI channel {mouse_name} - {area_name} with threshold {dapi_threshold}.')
            results['particles_rate'].append(0)
        else:
            results['particles_rate'].append(results[f'nb_particles_ieg'][-1] / results[f'nb_particles_dapi'][-1])

    return results


def particles_processing(
        data: NDArray,
        roi: NDArray,
        threshold: int,
        gaussian_sigma: float,
        min_particle_size: float,
        markers_percentile: float,
        show_plot: bool = True,
        save_plot_path: Path | str = None,
        silent: bool = False
) -> int:
    """
    Process the data to count the number of particles in a given ROI using watershed segmentation.

    :param data: The data to process, typically an image array.
    :param roi: The region of interest (ROI) mask to apply on the data.
    :param threshold: The detection threshold value for the pixels.
    :param gaussian_sigma: The sigma value for the Gaussian filter applied to the data before processing.
    :param min_particle_size: The minimum size of particles to be considered valid.
    :param markers_percentile: The percentile value used to generate markers for the watershed segmentation.
    :param show_plot: If True, display the plot of the data with the segmentation results.
    :param save_plot_path: If provided, save the plot to this path.
    :param silent: If True, suppress the output messages.

    :return: The number of particles detected in the ROI.
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
        print(f"Number of particles: {num_particles}")

    # Plot the data
    if show_plot or save_plot_path is not None:
        title = (f'Particles scan\nThreshold: {threshold}, Gaussian sigma: {gaussian_sigma}, '
                 f'Min particle size: {min_particle_size}, Markers percentile: {markers_percentile}')
        plot_data(data, roi, thr_mask, thr_and_roi_mask, labels, title=title, show_plot=show_plot,
                  save_path=save_plot_path)

    return num_particles


def save_all_particle_scans(
        data_dir: Path | str,
        out_dir: Path | str,
        threshold: int,
        gaussian_sigma: float,
        min_particle_size: float,
        markers_percentile: float,
        rolling_ball_radius: float | None = 90,
        mouse_filter: str = '*',
        area_filter: str = '*',
        use_cache: bool = True
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

    for roi, img_data, area_name, mouse_name in (
            batch_iterator(data_dir, mouse_filter, area_filter, ieg_channel=True, dapi_channel=False,
                           rolling_ball_radius=rolling_ball_radius, use_cache=use_cache)
    ):
        img = img_data["ieg"]
        file_name = Path(f'particles_{data_dir.name}_{mouse_name}_{area_name}.png')
        particles_processing(img, roi, threshold, gaussian_sigma, min_particle_size, markers_percentile,
                             show_plot=False, silent=True, save_plot_path=out_dir / file_name)
