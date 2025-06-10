import pytest

from commun import open_image, open_roi
from src.particles_analysis import particles_processing, particles_batch_processing


def test_typical_single_processing():
    # Given
    roi = open_roi('examples/crusI-left/Default/0982-1002.roi')
    threshold = 1150
    gaussian_sigma=1
    min_particle_size=30
    markers_percentile=90
    rolling_ball_radius=90

    # Channel 1
    data = open_image('examples/crusI-left/Default/img_channel001_position000_time000000000_z000.tif')
    result_channel_1 = particles_processing(
        data,
        roi,
        threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        rolling_ball_radius=rolling_ball_radius
    )

    assert result_channel_1 == pytest.approx(269, 1)

    # Channel 2
    data = open_image('examples/crusI-left/Default/img_channel002_position000_time000000000_z000.tif')
    result_channel_2 = particles_processing(
        data,
        roi,
        threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        rolling_ball_radius=rolling_ball_radius
    )

    assert result_channel_2 == pytest.approx(73, 1)
    assert result_channel_1 > result_channel_2
    assert result_channel_2 / result_channel_1 == pytest.approx(0.25, 0.03)

def test_batch_processing():
    # Given
    data_dir = 'examples'
    mouse_filter = 'mouse*'
    area_filter = 'L*'

    # When
    results = particles_batch_processing(
        data_dir, mouse_filter, area_filter,
        dapi_threshold=1624,
        ieg_threshold=800,
        gaussian_sigma=2,
        min_particle_size=30,
        markers_percentile=90,
        rolling_ball_radius=90
    )

    # Then
    assert len(results['mouse_name']) == 7
    assert len(results['area_name']) == 7
    assert len(results['particles_rate']) == 7