from pathlib import Path

import pytest

from area_analysis import area_processing, area_batch_processing
from commun import open_image, open_roi


def test_typical_single_processing():
    # Given
    data_path = Path('examples/crusI 20x-center/Default/img_channel002_position000_time000000000_z000.tif')
    roi_path = 'examples/crusI 20x-center/Default/1026-1015.roi'
    threshold = 1750
    rolling_ball_radius=90

    # When
    data = open_image(data_path)
    roi = open_roi(roi_path)
    result = area_processing(data, roi, threshold, rolling_ball_radius)

    # Then
    assert result == pytest.approx(0.44396, 0.01)

def test_batch_processing():
    # Given
    data_dir = 'examples'
    mouse_filter = 'mouse_A'
    area_filter = '*slice1*'
    threshold = 1750
    rolling_ball_radius=90

    # When
    results = area_batch_processing(data_dir, threshold, rolling_ball_radius, mouse_filter, area_filter)

    # Then
    assert len(results['mouse_name']) == 5
    assert len(results['area_name']) == 5
    assert len(results['roi_rate']) == 5