from logging import warning
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from numpy._typing import NDArray
from skimage import restoration
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
        rolling_ball_radius: float | None = 90
) -> Dict[str, List[Any]]:
    """
    Analyze all the data from the directories matching the filters.
    The folder structure is assumed to be : data_dir/mouse_dir/area_dir/Default
    With inside:
        - exactly one data file matching the channel number as "img_channel00N*"
        - exactly one ROI file matching "*.roi"

    :param data_dir: The relative or absolute path to the directory that contains all the data.
    :param threshold: The detection threshold value for the pixels.
    :param mouse_filter: The pattern to select the mouse subfolder to process. Where '*' is a wildcard.
    :param area_filter: The pattern to select the area subfolder to process. Where '*' is a wildcard.
    :param channel: The channel number for the data file.
    :return: The statistics as a dictionary where the keys are the field names and the values are lists of values.
    """
    # Prepare the result dictionary with empty lists
    results = {'mouse_name': [], 'area_name': [], 'nb_particles_dapi': [], 'nb_particles_ieg': [],
               'particles_rate': []}

    # Files iterator
    for roi, img_data, area_name, mouse_name in (
            batch_iterator(data_dir, mouse_filter, area_filter, ieg_channel=True, dapi_channel=True)
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
                rolling_ball_radius=rolling_ball_radius,
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
        rolling_ball_radius: float | None = None,
        show_plot: bool = True,
        save_plot_path: Path | str = None,
        silent: bool = False
) -> int:

    if rolling_ball_radius:
        # Apply rolling ball background subtraction
        background = restoration.rolling_ball(data, radius = rolling_ball_radius)
        # Subtract background
        processed_data = data - background
    else:
        processed_data = data

    # Apply Gaussian filter
    blurred = gaussian_filter(processed_data, sigma=gaussian_sigma)

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
        plot_data(data, processed_data, roi, thr_mask, thr_and_roi_mask, labels, title=title, show_plot=show_plot,
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
        area_filter: str = '*'
) -> None:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    for roi, img_data, area_name, mouse_name in (
            batch_iterator(data_dir, mouse_filter, area_filter, ieg_channel=True, dapi_channel=False)
    ):
        img = img_data["ieg"]
        file_name = Path(f'particles_{data_dir.name}_{mouse_name}_{area_name}.png')
        particles_processing(img, roi, threshold, gaussian_sigma, min_particle_size, markers_percentile,
                             rolling_ball_radius, show_plot=False, silent=True, save_plot_path=out_dir / file_name)


if __name__ == '__main__':
    # Meta-parameters
    dapi_threshold = 1624
    ieg_threshold = 800
    gaussian_sigma = 2
    min_particle_size = 30
    markers_percentile = 90
    rolling_ball_radius = 90  # Set to None to disable rolling ball background subtraction

    # Plot all the data
    save_all_particle_scans(
        data_dir='../data',
        out_dir='../out/particle_scans',
        threshold=ieg_threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        rolling_ball_radius=rolling_ball_radius,
        mouse_filter='*',
        area_filter='*lobule8*'
    )

    # Example usage for one image and one ROI
    example_data = open_image('data/L_CrusI_20x_center_left/Default/img_channel001_position000_time000000000_z000.tif')
    example_roi = open_roi('data/L_CrusI_20x_center_left/Default/1006-0970.roi')
    particles_processing(example_data, example_roi, ieg_threshold, gaussian_sigma, min_particle_size,
                         markers_percentile, rolling_ball_radius)

    # results_csv = particles_batch_processing(
    #     data_dir='../data',
    #     mouse_filter='*',
    #     area_filter='*',
    #     dapi_threshold=dapi_threshold,
    #     ieg_threshold=ieg_threshold,
    #     gaussian_sigma=gaussian_sigma,
    #     min_particle_size=min_particle_size,
    #     markers_percentile=markers_percentile,
    #     rolling_ball_radius=rolling_ball_radius
    # )
    #
    # save_results(out_directory='../out', file_name='particles_results.csv', results=results_csv,
    #              metadata={
    #                 'dapi_threshold': dapi_threshold,
    #                 'ieg_threshold': ieg_threshold,
    #                 'gaussian_sigma': gaussian_sigma,
    #                 'min_particle_size': min_particle_size,
    #                 'markers_percentile': markers_percentile,
    #                 'rolling_ball_radius': rolling_ball_radius
    #              })
