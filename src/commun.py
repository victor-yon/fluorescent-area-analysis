import csv
import json
from logging import warning
from pathlib import Path
from typing import Generator, Any

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
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


def save_results(
        out_directory: Path | str,
        file_name: str,
        results: dict[str, Any],
        metadata: dict[str, Any] = None
) -> None:
    if isinstance(out_directory, str):
        out_directory = Path(out_directory)

    if not file_name.lower().endswith('.csv'):
        file_name += '.csv'

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

    if metadata is not None:
        with open(file_path.with_suffix('_metadata.json'), 'w') as file:
            json.dump(metadata, file, indent=4)

    print(f'Result saved in "{file_path.resolve()}"')


def batch_iterator(
        data_dir: Path | str,
        mouse_filter: str = '*',
        area_filter: str = '*',
        dapi_channel: bool = True,
        ieg_channel: bool = True
) -> Generator[tuple[NDArray, dict[str, NDArray], str, str], None, None]:
    """
    Open all the data from the directories matching the filters.
    The folder structure is assumed to be : data_dir/mouse_dir/area_dir/Default
    With inside:
        - exactly one data file matching the channel number as "img_channel00N*"
        - exactly one ROI file matching "*.roi"

    :param data_dir: The relative or absolute path to the directory that contains all the data.
    :param mouse_filter: The pattern to select the mouse subfolder to process. Where '*' is a wildcard.
    :param area_filter: The pattern to select the area subfolder to process. Where '*' is a wildcard.
    :param dapi_channel: If True, process the DAPI channel (number 1).
    :param ieg_channel: If True, process the IEG channel (number 2).
    :param dapi_threshold: The detection threshold value for the DAPI pixels.
    :param ieg_threshold: The detection threshold value for the IEG pixels.
    :return: A tuple containing the ROI data and a dictionary with the image data for each channel.
    """

    if not dapi_channel and not ieg_channel:
        raise ValueError('At least one channel must be selected (DAPI or IEG).')

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f'Invalid data directory: "{data_dir.resolve()}".')

    mouse_directories = list(data_dir.glob(mouse_filter))
    nb_mouse_found = len(mouse_directories)

    if nb_mouse_found == 0:
        raise FileNotFoundError(f'No mouse subfolder found in "{data_dir.resolve()}" with filter "{mouse_filter}".')
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

            channel_list = []
            if dapi_channel:
                channel_list.append(('dapi', 1))
            if ieg_channel:
                channel_list.append(('ieg', 2))

            img_data = {}
            for name, channel in channel_list:
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

                img_data[name] = open_image(data_path)

            yield roi, img_data, area_name, mouse_directory.stem

            nb_area_processed += 1

        if nb_area_processed == nb_area_found:
            print(f'The {nb_area_processed} folder(s) matching "{area_filter}" in "{mouse_directory.stem}" have been '
                  f'successfully processed.')
        else:
            print(f'{nb_area_processed}/{nb_area_found} folder(s) matching "{area_filter}" in "{mouse_directory.stem}" '
                  f'have been successfully processed.')
            warning(f'{nb_area_found - nb_area_processed} folder(s) skipped because the expected files could not be '
                    f'automatically detected.')
