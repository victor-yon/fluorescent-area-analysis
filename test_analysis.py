import csv
from logging import warning
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from numpy._typing import NDArray
from roifile import ImagejRoi


def open_image(path: Path | str) -> NDArray:
    """
    Open a TIF image as a numpy array.

    :param path: The path to the ".tif" file.
    :return A numpy array representing the image.
    """
    img = Image.open(path)
    return np.array(img)


def open_roi(path: Path | str) -> NDArray:
    """
    Open a Region Of Interest (ROI) file generated with ImageJ and convert it to a Python friendly format.
    Use "roifile" package to process the file: https://doi.org/10.5281/zenodo.6941603

    :param path: The path to the ".roi" file.
    :return: The ROI coordinates as a numpy array.
    """
    return ImagejRoi.fromfile(path).coordinates()


def get_threshold_mask(img_array: NDArray, threshold: int) -> NDArray[bool]:
    """
    Create a mask of pixels above a given threshold.

    :param img_array: The image as a numpy array.
    :param threshold: The threshold value.
    :return: A threshold mask as a matrice of boolean.
    """
    return img_array > threshold


def get_roi_mask(img_array: NDArray, roi_coordinates: NDArray) -> NDArray[bool]:
    """
    Create a mask of pixels inside the ROI.

    :param img_array: The image as a numpy array.
    :param roi_coordinates: The ROI coordinates.
    :return: A ROI mask as a matrice of boolean.
    """
    # Get the image dimensions
    img_height, img_width = img_array.shape[:2]
    # Create a mask image with the same dimensions as the input image
    mask_img = Image.new('L', (img_width, img_height), 0)
    # Create a drawing context
    draw = ImageDraw.Draw(mask_img)
    # Convert coordinates to a list of tuples
    polygon = [(x, y) for x, y in roi_coordinates]
    # Draw the polygon on the mask image
    draw.polygon(polygon, outline=1, fill=1)
    # Convert the mask image to a numpy array
    mask = np.array(mask_img).astype(bool)
    return mask


def batch_processing(data_dir: Path | str, threshold: int, mouse_filter: str = '*', area_filter: str = '*',
                     channel: int = 2) -> Dict[str, List[Any]]:
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
    results = {'mouse_name': [], 'area_name': [], 'roi_rate': []}

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

            # Search for the data file
            data_file_pattern = f'img_channel{channel:03}*'
            data_path = list(area_directory.glob(data_file_pattern))
            if len(data_path) == 1:
                data_path = data_path[0]
            else:
                warning(f'Data file not found in "{area_directory.resolve()}", '
                        f'with pattern "{data_file_pattern}". Expect 1 but got {len(data_path)}. '
                        f'This experiment is skipped.')
                continue

            img = open_image(data_path)
            roi = open_roi(roi_path)

            # Create masks and combine them
            thr_mask = get_threshold_mask(img, threshold)
            roi_mask = get_roi_mask(img, roi)
            thr_and_roi_mask = np.logical_and(thr_mask, roi_mask)

            # Compute statistics
            results['mouse_name'].append(mouse_directory.stem)
            results['area_name'].append(area_name)
            results['roi_rate'].append(thr_and_roi_mask.sum() / roi_mask.sum())
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


def save_results(out_directory: Path | str, file_name: str, results) -> None:
    if isinstance(out_directory, str):
        out_directory = Path(out_directory)
    # Writing to CSV file
    out_directory.mkdir(parents=True, exist_ok=True)
    file_path = out_directory / file_name
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(results.keys())
        # Write data
        rows = zip(*results.values())
        writer.writerows(rows)

    print(f'Result saved in "{file_path.resolve()}"')


def plot_data(data: NDArray, roi: NDArray, thr_mask: NDArray[bool], thr_and_roi_mask: NDArray[bool]) -> None:
    """
    Plot the original image, the original image with ROI, the threshold mask, and the combined threshold & ROI mask.

    :param data: Original image data as a numpy array.
    :param roi: ROI coordinates as a numpy array.
    :param thr_mask: Threshold mask as a numpy array.
    :param thr_and_roi_mask: Combined threshold and ROI mask as a numpy array.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1. Original Image
    axes[0, 0].imshow(data, cmap='seismic')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # 2. Original Image with ROI
    axes[0, 1].imshow(data, cmap='seismic')
    axes[0, 1].set_title("Original Image with ROI")
    roi_polygon = plt.Polygon(roi, fill=None, edgecolor='r', linewidth=2)
    axes[0, 1].add_patch(roi_polygon)
    axes[0, 1].axis('off')

    # 3. Threshold Mask
    axes[1, 0].imshow(thr_mask, cmap='gray')
    axes[1, 0].set_title("Threshold Mask")
    axes[1, 0].axis('off')

    # 4. Combined Threshold & ROI Mask
    axes[1, 1].imshow(thr_and_roi_mask, cmap='gray')
    axes[1, 1].set_title("Threshold + ROI Mask")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def single_processing(data_file: Path | str, roi_file: Path | str, threshold: int) -> float:
    """
    Simple function to evaluate specific data and ROI files.

    :param data_file: The path to the TIF image file to process.
    :param roi_file: The path to the ROI file to process.
    :param threshold: The detection threshold value.
    :return: The rate of pixel above the thoreshold inside the ROI.
    """
    # Open and process the data and ROI files
    data = open_image(data_file)
    roi = open_roi(roi_file)

    # Compute the masks
    thr_mask = get_threshold_mask(data, threshold)
    roi_mask = get_roi_mask(data, roi)
    thr_and_roi_mask = np.logical_and(thr_mask, roi_mask)

    # Plot the data
    plot_data(data, roi, thr_mask, thr_and_roi_mask)

    # Calculate the rate of pixel above the threshold inside the ROI
    roi_rate = thr_and_roi_mask.sum() / roi_mask.sum()
    print(f'ROI rate: {roi_rate:.4%}')

    return roi_rate


if __name__ == '__main__':
    # single_processing(data_file='data/test.tif', roi_file='data/test.roi', threshold=1050)
    results_csv = batch_processing(data_dir='data/sample',
                                   mouse_filter='mouse1',
                                   area_filter='*',
                                   threshold=1050,
                                   channel=2)
    save_results(out_directory='out', file_name='results.csv', results=results_csv)