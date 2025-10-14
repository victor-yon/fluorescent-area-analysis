from pathlib import Path

from src.commun import open_image, open_roi
from src.particles_analysis import (
    particles_batch_processing,
    particles_processing_gaussian_laplace,
    particles_processing_overlapping,
    particles_processing_threshold,
)


def test_simple_single_processing():
    # Given
    path = Path("tests/examples/crusI-left/Default")
    roi = open_roi("tests/examples/crusI-left/Default/0982-1002.roi")
    threshold = 1150
    gaussian_sigma = 1
    min_particle_size = 30
    markers_percentile = 90

    # Channel DAPI
    data = open_image(path, 1)
    result_channel_dapi, labels = particles_processing_threshold(
        data,
        roi,
        threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        show_plot=True,
        silent=True,
    )

    assert result_channel_dapi > 0

    # Channel IEG
    data = open_image(path, 2)

    result_channel_ieg_overlapping = particles_processing_overlapping(
        data,
        gaussian_sigma,
        labels,
        threshold,
        show_plot=False,
        silent=True,
    )

    assert result_channel_ieg_overlapping > 0

    result_channel_ieg_threshold, _ = particles_processing_threshold(
        data,
        roi,
        threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        show_plot=False,
        silent=True,
    )

    assert result_channel_ieg_threshold > 0

    result_channel_ieg_gaussian_laplace = particles_processing_gaussian_laplace(
        data,
        roi,
        gaussian_sigma=gaussian_sigma,
        labels=labels,
        show_plot=False,
        silent=True,
    )

    assert result_channel_ieg_gaussian_laplace > 0

    print(
        f"Results: \n\t- DAPI={result_channel_dapi}"
        f"\n\t- IEG overlapping={result_channel_ieg_overlapping}"
        f"\n\t- IEG threshold (legacy)={result_channel_ieg_threshold}"
        f"\n\t- IEG Gaussian Laplace={result_channel_ieg_gaussian_laplace}"
    )


def test_batch_processing():
    # Given
    data_dir = "examples"
    mouse_filter = "mouse*"
    area_filter = "L*"

    # When
    results = particles_batch_processing(
        data_dir,
        mouse_filter,
        area_filter,
        dapi_threshold=1624,
        ieg_threshold=800,
        gaussian_sigma=2,
        min_particle_size=30,
        markers_percentile=90,
        rolling_ball_radius=None,
    )

    # Then
    assert len(results["mouse_name"]) == 7
    assert len(results["area_name"]) == 7
    assert len(results["particles_rate"]) == 7
