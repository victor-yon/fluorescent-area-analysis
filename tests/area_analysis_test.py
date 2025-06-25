from pathlib import Path

import pytest

from area_analysis import area_processing, area_batch_processing
from commun import open_image, open_roi


def test_simple_single_processing():
    # Given
    dir_path = Path('examples/crusI 20x-center/Default')
    roi_path = 'examples/crusI 20x-center/Default/1026-1015.roi'
    threshold = 1750

    # When
    data = open_image(dir_path, 2)
    roi = open_roi(roi_path)
    result = area_processing(data, roi, threshold, show_plot=False, silent=True)

    # Then
    assert result == pytest.approx(0.44396, 0.01)


def test_batch_processing():
    # Given
    data_dir = 'examples'
    mouse_filter = 'mouse_A'
    area_filter = '*slice1*'
    threshold = 1750

    # When
    results = area_batch_processing(data_dir, threshold, None, mouse_filter, area_filter)

    # Then
    assert len(results['mouse_name']) == 5
    assert len(results['area_name']) == 5
    assert len(results['roi_rate']) == 5