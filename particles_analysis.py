from logging import warning
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from scipy.ndimage import gaussian_filter, distance_transform_edt

from commun import open_image, open_roi, get_threshold_mask, get_roi_mask, plot_data, save_results


def batch_processing(data_dir: Path | str, mouse_filter: str = '*', area_filter: str = '*',
                     dapi_channel: int = 1, ieg_channel: int = 2, dapi_threshold=1624, ieg_threshold=800,
                     gaussian_sigma=2, min_particule_size=30, markers_percentile=90) -> Dict[str, List[Any]]:
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
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f'Invalid data directory: "{data_dir.resolve()}".')

    # Prepare the result dictionary with empty listes
    results = {'mouse_name': [], 'area_name': [], 'nb_particules_dapi': [], 'nb_particules_ieg': [],
               'particules_rate': []}

    mouse_directories = list(data_dir.glob(mouse_filter))
    nb_mouse_found = len(mouse_directories)

    if nb_mouse_found == 0:
        raise RuntimeError(f'No mouse subfolder found in "{data_dir.resolve()}" with filter "{mouse_filter}".')
    elif nb_mouse_found == 1:
        print(f'1 mouse subfolder found in the directory "{data_dir.resolve()}".')
    else:
        print(f'{nb_mouse_found} mouse subfolders found in the "{data_dir.resolve()}" matching "{mouse_filter}".')

    # Iterate mouse directory
    for mouse_directory in mouse_directories:
        # Search for subfolders that match the area filter
        area_directories = list(mouse_directory.glob(area_filter))
        nb_area_found = len(area_directories)
        nb_area_processed = 0

        if nb_area_found == 0:
            raise RuntimeError(f'No area subfolder found in "{mouse_directory.resolve()}" with filter "{area_filter}".')
        elif nb_area_found == 1:
            print(f'1 area subfolder found in the directory "{mouse_directory.resolve()}".')
        else:
            print(f'{nb_area_found} area subfolders found in the "{mouse_directory.resolve()}"'
                  f' matching "{area_filter}".')

        # Iterate experiment directory
        for i, area_directory in enumerate(area_directories, 1):
            area_name = area_directory.stem
            print(f'{i:03}/{nb_area_found:03} - {i / nb_area_found:>6.1%}: {mouse_directory.stem} - {area_name}')

            area_directory = area_directory / 'Default'

            # Search for the ROI file
            roi_path = list(area_directory.glob('*.roi'))
            if len(roi_path) == 1:
                roi_path = roi_path[0]
            else:
                warning(f'ROI file not found in "{area_directory.resolve()}". Expect 1 but got {len(roi_path)}. '
                        f'This experiment is skipped.')
                continue

            roi = open_roi(roi_path)

            for name, channel, threshold in [('dapi', dapi_channel, dapi_threshold), ('ieg', ieg_channel, ieg_threshold)]:
                # Search for the data file
                data_file_pattern = f'img_channel{channel:03}*'
                data_path = list(area_directory.glob(data_file_pattern))
                if len(data_path) == 1:
                    data_path = data_path[0]
                else:
                    warning(f'Data file "{name}" not found in "{area_directory.resolve()}", '
                            f'with pattern "{data_file_pattern}". Expect 1 but got {len(data_path)}. '
                            f'This experiment is skipped.')
                    continue

                img = open_image(data_path)

                # Apply Gaussian filter
                blurred = gaussian_filter(img, sigma=gaussian_sigma)

                # Create masks and combine them
                thr_mask = get_threshold_mask(blurred, threshold)
                roi_mask = get_roi_mask(blurred, roi)
                thr_and_roi_mask = np.logical_and(thr_mask, roi_mask)

                # Compute the distance transform
                distance = distance_transform_edt(thr_and_roi_mask)

                # Generate markers, with Nth percentile for strong peaks
                markers = label(distance > np.percentile(distance, markers_percentile))

                # Apply watershed segmentation
                labels = watershed(-distance, markers, mask=thr_and_roi_mask)
                labels = remove_small_objects(labels, min_size=min_particule_size)

                # Count the number of particles
                num_particles = len(np.unique(labels)) - 1

                results[f'nb_particules_{name}'].append(num_particles)

            # Compute statistics
            results['mouse_name'].append(mouse_directory.stem)
            results['area_name'].append(area_name)
            results['particules_rate'].append(results[f'nb_particules_ieg'][-1] / results[f'nb_particules_dapi'][-1])
            nb_area_processed += 1

        if nb_area_processed == nb_area_found:
            print(f'The {nb_area_processed} folder(s) matching "{area_filter}" in "{mouse_directory.stem}" have been '
                  f'successfully processed.')
        else:
            print(f'{nb_area_processed}/{nb_area_found} folder(s) matching "{area_filter}" in "{mouse_directory.stem}" '
                  f'have been successfully processed.')
            warning(f'{nb_area_found - nb_area_processed} folder(s) skipped because the expected files could not be '
                    f'automatically detected.')

    return results


def single_processing(data_file: str, roi_file: str, threshold: int, gaussian_sigma: float,
                      min_particule_size: float, markers_percentile: float) -> int:
    # Open files
    data = open_image(data_file)
    roi = open_roi(roi_file)

    # Apply Gaussian filter
    blurred = gaussian_filter(data, sigma=gaussian_sigma)

    # Apply thresholding and ROI mask
    thr_mask = get_threshold_mask(blurred, threshold)
    roi_mask = get_roi_mask(blurred, roi)
    thr_and_roi_mask = np.logical_and(thr_mask, roi_mask)

    # Compute the distance transform
    distance = distance_transform_edt(thr_and_roi_mask)

    # Generate markers, with Nth percentile for strong peaks
    markers = label(distance > np.percentile(distance, markers_percentile))

    # Apply watershed segmentation
    labels = watershed(-distance, markers, mask=thr_and_roi_mask)
    labels = remove_small_objects(labels, min_size=min_particule_size)

    # Count the number of particles
    num_particles = len(np.unique(labels)) - 1

    # Plot the data
    plot_data(data, roi, thr_mask, thr_and_roi_mask, labels)

    print(f"Number of particles: {num_particles}")

    return num_particles


if __name__ == '__main__':
    # single_processing(
    #     data_file='data/L_CrusI_20x_center_left/Default/img_channel001_position000_time000000000_z000.tif',
    #     roi_file="data/L_CrusI_20x_center_left/Default/1006-0970.roi", threshold=1624,
    #     gaussian_sigma=2, min_particule_size=30, markers_percentile=90)

    results_csv = batch_processing(data_dir='data/Aug 25 2024 rotarod 20x', mouse_filter='89887-1*', area_filter='*lobule*',
                     dapi_channel=1, ieg_channel=2, dapi_threshold=1624, ieg_threshold=800,
                     gaussian_sigma=2, min_particule_size=30, markers_percentile=90)

    save_results(out_directory='out', file_name='particules_results.csv', results=results_csv)
