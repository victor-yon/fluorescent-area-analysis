from pathlib import Path

from src.commun import open_image, open_roi
from src.particles_analysis import (
    particles_batch_processing,
    particles_processing_dapi,
    particles_processing_ieg,
)


def test_simple_single_processing():
    # Given
    path = Path("tests/examples/crusI-left/Default")
    roi = open_roi("tests/examples/crusI-left/Default/0982-1002.roi")
    threshold = 1150
    gaussian_sigma = 1
    min_particle_size = 30
    markers_percentile = 90

    # Channel 1
    data = open_image(path, 1)
    result_channel_dapi, labels = particles_processing_dapi(
        data,
        roi,
        threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        show_plot=False,
        silent=True,
    )

    assert result_channel_dapi > 0

    # Channel 2
    data = open_image(path, 2)
    result_channel_ieg, _ = particles_processing_dapi(
        data,
        roi,
        threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        show_plot=False,
        silent=True,
    )

    assert result_channel_ieg > 0

    print(
        f"Legacy method result: DAPI: {result_channel_dapi:,d}, IEG: {result_channel_ieg:,d}"
        f" Ratio: {result_channel_ieg / result_channel_dapi:.2f}"
    )

    result_channel_ieg = particles_processing_ieg(
        data,
        roi,
        gaussian_sigma=gaussian_sigma,
        labels=labels,
        show_plot=False,
        silent=True,
    )

    print(
        f"New method result: DAPI: {result_channel_dapi:,d}, IEG: {result_channel_ieg:,d}"
        f" Ratio: {result_channel_ieg / result_channel_dapi:.2f}"
    )

    assert False

    # assert result_channel_2 == pytest.approx(73, 1)
    # assert result_channel_1 > result_channel_2
    # assert result_channel_2 / result_channel_1 == pytest.approx(0.25, 0.03)


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
