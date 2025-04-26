from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from numpy._typing import NDArray

from commun import open_image, open_roi, get_threshold_mask, get_roi_mask, save_results, plot_data, batch_iterator


def area_batch_processing(
        data_dir: Path | str,
        threshold: int,
        mouse_filter: str = '*',
        area_filter: str = '*'
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
    results = {'mouse_name': [], 'area_name': [], 'roi_rate': []}

    for roi, img_data, area_name, mouse_name in (
            batch_iterator(data_dir, mouse_filter, area_filter, ieg_channel=True)
    ):
        img = img_data["ieg"]
        roi_rate = area_processing(img, roi, threshold, show_plot=False, silent=True)

        # Compute statistics
        results['mouse_name'].append(mouse_name)
        results['area_name'].append(area_name)
        results['roi_rate'].append(roi_rate)

    return results


def area_processing(
        data: NDArray,
        roi: NDArray,
        threshold: int,
        show_plot: bool = True,
        silent: bool = False
) -> float:
    """
    Simple function to evaluate specific data and ROI files.

    :param data: The data to process.
    :param roi: The ROI to process.
    :param threshold: The detection threshold value.
    :return: The rate of pixel above the thoreshold inside the ROI.
    """
    # Compute the masks
    thr_mask = get_threshold_mask(data, threshold)
    roi_mask = get_roi_mask(data, roi)
    thr_and_roi_mask = np.logical_and(thr_mask, roi_mask)

    # Plot the data
    if show_plot:
        plot_data(data, roi, thr_mask, thr_and_roi_mask)

    # Calculate the rate of pixel above the threshold inside the ROI
    roi_rate = thr_and_roi_mask.sum() / roi_mask.sum()

    if not silent:
        print(f'ROI rate: {roi_rate:.4%}')

    return roi_rate


if __name__ == '__main__':
    # Example usage for one image and one ROI
    example_data = open_image('tests/examples/crusI-left/Default/img_channel002_position000_time000000000_z000.tif')
    example_roi = open_roi('tests/examples/crusI-left/Default/0982-1002.roi')
    area_processing(example_data, example_roi, threshold=1050)

    # Example usage for processing multiple images and ROIs
    # results_csv = area_batch_processing(
    #     data_dir='data/sample',
    #     mouse_filter='mouse1',
    #     area_filter='*',
    #     threshold=1050
    # )
    # save_results(out_directory='out', file_name='results.csv', results=results_csv)
