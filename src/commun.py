import csv
from logging import warning
from pathlib import Path
from typing import Generator, Any

import numpy as np
from PIL import Image, ImageDraw
from numpy._typing import NDArray
from roifile import ImagejRoi
from skimage import restoration


def open_image(
        directory: Path | str,
        channel: int,
        rolling_ball_radius: float | None = None,
        use_cache: bool = True
) -> NDArray:
    """
    Open a TIF image as a numpy array.

    :param directory: The path of the directory that contains the ".tif" file.
    :param channel: The channel number to open (1 for DAPI, 2 for IEG).
    :param rolling_ball_radius: The radius for the rolling ball background subtraction for pre-processing.
        If None, no background subtraction is applied.
    :param use_cache: If True, the processed image is cached to avoid slow reprocessing.

    :return A numpy array representing the image.
    """

    if isinstance(directory, str):
        directory = Path(directory)

    if rolling_ball_radius is not None and use_cache:
        cache_file = directory / f'img_channel{channel:03}_processed_{rolling_ball_radius:03}.npy'
        # Check if the processed image is already cached
        if cache_file.exists():
            return np.load(cache_file)
    else:
        cache_file = None

    # Search for the data file
    data_file_pattern = f'img_channel{channel:03}*.tif'
    data_path = list(directory.glob(data_file_pattern))
    if len(data_path) == 1:
        data_path = data_path[0]
    elif len(data_path) == 0:
        raise FileNotFoundError(
            f'Data file not found in "{directory.resolve()}" for channel {channel}.'
            f'Expected 1 but got 0.'
        )
    else:
        raise FileNotFoundError(
            f'Too many file found in "{directory.resolve()}" for channel {channel}.'
            f'Expect 1 but got {len(data_path)}.'
        )

    # Load ImageJ image as a numpy array
    data = np.array(Image.open(data_path))

    if rolling_ball_radius:
        # Apply rolling ball background subtraction
        background = restoration.rolling_ball(data, radius = rolling_ball_radius)
        # Subtract background
        data = data - background

        if use_cache:
            # Save the processed image to cache
            np.save(cache_file, data)

    return data


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

        if metadata is not None:
            # Write metadata as a comment
            for key, value in metadata.items():
                writer.writerow([f'# {key}: {value}'])

        # Write header
        writer.writerow(results.keys())
        # Write data
        rows = zip(*results.values())
        writer.writerows(rows)

    print(f'Result saved in "{file_path.resolve()}"')


def batch_iterator(
        data_dir: Path | str,
        mouse_filter: str = '*',
        area_filter: str = '*',
        dapi_channel: bool = True,
        ieg_channel: bool = True,
        rolling_ball_radius: float | None = None,
        use_cache: bool = True
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
    :param rolling_ball_radius: The radius for the rolling ball background subtraction for pre-processing.
       If None, no background subtraction is applied.
    :param use_cache: If True, the processed image is loaded or saved from cached to avoid slow reprocessing.
    :return: A tuple containing the ROI data and a dictionary with the image data for each channel.
    """

    if not dapi_channel and not ieg_channel:
        raise ValueError('At least one channel must be selected (DAPI or IEG).')

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f'Invalid data directory: "{data_dir.resolve()}".')

    mouse_directories = [d for d in data_dir.glob(mouse_filter) if d.is_dir()]
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
            warning(f'No area subfolder found in "{mouse_directory.resolve()}" with filter "{area_filter}".')
        elif nb_area_found == 1:
            print(f'1 area subfolder found in the directory "{mouse_directory.resolve()}".')
        else:
            print(f'{nb_area_found} area subfolders found in the "{mouse_directory.resolve()}"'
                  f' matching "{area_filter}".')

        # Iterate experiment directory
        for i, area_directory in enumerate(area_directories, 1):
            area_name = area_directory.stem
            print(f'{i:03}/{nb_area_found:03} - {i / nb_area_found:>6.1%}: {mouse_directory.stem} - {area_name}')

            files_directory = area_directory / 'Default'

            # Try no capitalization for "Default" folder
            if not files_directory.exists():
                files_directory = area_directory / 'default'

            if not files_directory.exists():
                warning(f'No "Default" subfolder found in "{area_directory.resolve()}". This experiment is skipped.')
                continue

            # Search for the ROI file
            roi_path = list(files_directory.glob('*.roi'))
            if len(roi_path) == 1:
                roi_path = roi_path[0]
            else:
                warning(f'ROI file not found in "{files_directory.resolve()}". Expect 1 but got {len(roi_path)}. '
                        f'This experiment is skipped.')
                continue

            roi = open_roi(roi_path)

            channel_list = []
            if dapi_channel:
                channel_list.append(('dapi', 1))
            if ieg_channel:
                channel_list.append(('ieg', 2))

            img_data = {}
            try:
                for name, channel in channel_list:
                    img_data[name] = open_image(files_directory, channel, rolling_ball_radius, use_cache)
            except FileNotFoundError as e:
                warning(f'Error with image loading in "{files_directory.resolve()}":\n{e}')
                continue

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
