import matplotlib.pyplot as plt
import numpy as np
from skimage import io, restoration

# Load image
image = io.imread(r"D:\Dec 8 2024 rotarod GCL\91038-1 WT M rotarod\L_slice1_crusI-center\Default\img_channel002_position000_time000000000_z000.tif")

# Define rolling ball radius
radius = 90

# Apply rolling ball background subtraction
background = restoration.rolling_ball(image, radius = radius)

# Subtract background
result = image - background

# === Set brightness/contrast manually ===
vmin = 0
vmax = 2000  # Try lowering this if the image looks dark; increase if it's too washed out

def plot_result(image, background, result):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    ax[0].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title('Background')
    ax[1].axis('off')

    ax[2].imshow(result, cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()
    plt.show()

plot_result(image, background, result)
