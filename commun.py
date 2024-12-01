import csv
from pathlib import Path

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


def plot_data(data: NDArray, roi: NDArray, thr_mask: NDArray[bool],
              thr_and_roi_mask: NDArray[bool] = None, particules_labels: NDArray[bool] = None) -> None:
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
    if thr_and_roi_mask is not None:
        axes[1, 1].imshow(thr_and_roi_mask, cmap='gray')
        axes[1, 1].set_title("Threshold + ROI Mask")
        axes[1, 1].axis('off')

    # 4. Combined Threshold & ROI Mask with Particules
    if particules_labels is not None:
        axes[1, 1].imshow(particules_labels, cmap='prism', alpha=0.5)
        axes[1, 1].set_title("Particules")
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
