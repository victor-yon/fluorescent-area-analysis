from pathlib import Path

from src.commun import open_image


def test_single_file_with_rolling_ball():
    # Given
    dir_path = Path("examples/crusI 20x-center/Default")
    rolling_ball_radius = 3  # Small radius to accelerate the test

    # When
    data = open_image(dir_path, 2, rolling_ball_radius, use_cache=False)

    # Then
    assert data is not None


def test_single_file_with_rolling_ball_cached():
    # Given
    dir_path = Path("examples/crusI 20x-center/Default")
    rolling_ball_radius = 90

    # Should load from cache
    data = open_image(dir_path, 2, rolling_ball_radius, use_cache=True)

    # Then
    assert data is not None
