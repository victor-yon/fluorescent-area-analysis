from logging import warning, error
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from joblib import Parallel, delayed
from numpy._typing import NDArray

from commun import open_image, open_roi, get_threshold_mask, get_roi_mask, batch_iterator, save_results
from plots import plot_data


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
    def job_wrapper(roi, img_data, area_name, mouse_name):
        """
        Wrapper function to process each image and ROI in parallel.
        """
        roi_rate = area_processing(img_data["ieg"], roi, threshold, show_plot=False, silent=True)
        return roi_rate, area_name, mouse_name

    # One job per CPU core - 1
    results = Parallel(n_jobs=-2, backend='threading')(
        delayed(job_wrapper)(roi, img_data, area_name, mouse_name) for
        roi, img_data, area_name, mouse_name in
        batch_iterator(data_dir, mouse_filter, area_filter, ieg_channel=True)
    )

    # Create the result dictionary
    results_dict = {'area_name': [], 'mouse_name': [], 'roi_rate': []}
    for rate, area_name, mouse_name in results:
        results_dict['area_name'].append(area_name)
        results_dict['mouse_name'].append(mouse_name)
        results_dict['roi_rate'].append(rate)

    return results_dict


def area_processing(
        data: NDArray,
        roi: NDArray,
        threshold: int,
        show_plot: bool = True,
        save_plot_path: Path | str = None,
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

    # Calculate the rate of pixel above the threshold inside the ROI
    if roi_mask.sum() == 0:
        roi_rate = 0
        error(f'ROI mask is empty.')
    else:
        roi_rate = thr_and_roi_mask.sum() / roi_mask.sum()

    if not silent:
        print(f'ROI rate: {roi_rate:.4%}')

    # Plot the data
    if show_plot or save_plot_path is not None:
        title = f'Area scan\nThreshold: {threshold}'
        plot_data(data, roi, thr_mask, thr_and_roi_mask, title=title, show_plot=show_plot, save_path=save_plot_path)

    return roi_rate

def save_all_area_scans(
        data_dir: Path | str,
        out_dir: Path | str,
        threshold: int,
        mouse_filter: str = '*',
        area_filter: str = '*'
) -> None:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    for roi, img_data, area_name, mouse_name in (
            batch_iterator(data_dir, mouse_filter, area_filter, ieg_channel=True, dapi_channel=False)
    ):
        img = img_data["ieg"]
        file_name = Path(f'area_{data_dir.name}_{mouse_name}_{area_name}.png')
        area_processing(img, roi, threshold, show_plot=False, silent=True, save_plot_path=out_dir / file_name)


if __name__ == '__main__':
    # Meta-parameters
    threshold = 1050

    # Plot all the data
    save_all_area_scans(
        data_dir='../data',
        out_dir='../out/area_scans',
        threshold=threshold,
        mouse_filter='*',
        area_filter='*lobule8*'
    )

    # Example usage for one image and one ROI
    # example_data = open_image('tests/examples/crusI-left/Default/img_channel002_position000_time000000000_z000.tif')
    # example_roi = open_roi('tests/examples/crusI-left/Default/0982-1002.roi')
    # area_processing(example_data, example_roi, threshold=1050)

    # Example usage for processing multiple images and ROIs
    # results_csv = area_batch_processing(
    #     data_dir='../data',
    #     mouse_filter='*',
    #     area_filter='*',
    #     threshold=threshold
    # )
    # save_results(out_directory='../out', file_name='results.csv', results=results_csv,
    #              metadata={'threshold': threshold})
