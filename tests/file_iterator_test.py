import pytest

from commun import batch_iterator


def test_iterate_specific_mouse():
    results = list(batch_iterator('examples', 'mouse_A'))
    assert len(results) == 8

def test_iterate_mice_filter():
    results = list(batch_iterator('examples', 'mouse*'))
    assert len(results) == 13

def test_iterate_area_filter():
    results = list(batch_iterator('examples', 'mouse_B', '*slice2*'))
    assert len(results) == 2

def test_iterate_mice_and_area_filter():
    results = list(batch_iterator('examples', 'mouse*', 'L*'))
    assert len(results) == 7

def test_iterate_missing_folder():
    with pytest.raises(FileNotFoundError):
        list(batch_iterator('wrong_directory'))

def test_one_channel():
    results = list(batch_iterator('examples', 'mouse_A', ieg_channel=True, dapi_channel=False))
    assert len(results[0]) == 4
    roi, img_data, area_name, mouse_directory = results[0]
    assert img_data['ieg'] is not None
    assert 'dapi' not in img_data
    assert roi is not None
    assert len(area_name) > 0
    assert len(mouse_directory) > 0

def test_two_channels():
    results = list(batch_iterator('examples', 'mouse_B', ieg_channel=True, dapi_channel=True))
    assert len(results[0]) == 4
    roi, img_data, area_name, mouse_directory = results[0]
    assert img_data['ieg'] is not None
    assert img_data['dapi'] is not None
    assert roi is not None
    assert len(area_name) > 0
    assert len(mouse_directory) > 0
