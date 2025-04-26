from logging import warning
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from scipy.ndimage import gaussian_filter, distance_transform_edt

from commun import open_image, open_roi, get_threshold_mask, get_roi_mask, plot_data, save_results, batch_iterator


def particles_batch_processing(
        data_dir: Path | str,
        mouse_filter: str = '*',
        area_filter: str = '*',
        dapi_threshold=1624,
        ieg_threshold=800,
        gaussian_sigma=2,
        min_particle_size=30,
        markers_percentile=90
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
        silent: bool = False
) -> int:
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

    # Plot the data
    if show_plot:
        plot_data(data, roi, thr_mask, thr_and_roi_mask, labels)

    if not silent:
        print(f"Number of particles: {num_particles}")

    return num_particles


if __name__ == '__main__':
    # Example usage for one image and one ROI
    example_data = open_image('data/L_CrusI_20x_center_left/Default/img_channel001_position000_time000000000_z000.tif')
    example_roi = open_roi('data/L_CrusI_20x_center_left/Default/1006-0970.roi')
    particles_processing(example_data, example_roi, threshold=1624,
                         gaussian_sigma=2, min_particle_size=30, markers_percentile=90)

    results_csv = particles_batch_processing(
        data_dir='data/Aug 25 2024 rotarod 20x',
        mouse_filter='89887-1*',
        area_filter='*lobule*',
        dapi_threshold=1624,
        ieg_threshold=800,
        gaussian_sigma=2,
        min_particle_size=30,
        markers_percentile=90
    )

    save_results(out_directory='out', file_name='particles_results.csv', results=results_csv)
