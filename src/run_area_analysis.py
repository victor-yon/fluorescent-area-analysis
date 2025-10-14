from src.area_analysis import (
    area_batch_processing,
    area_processing,
    save_all_area_scans,
)
from src.commun import open_image, open_roi, save_results

if __name__ == "__main__":
    # Meta-parameters
    threshold = 1050
    rolling_ball_radius = (
        90  # Set to None to disable rolling ball background subtraction
    )

    # ====================================================================
    # Example usage for one image and one ROI
    # ====================================================================

    example_data = open_image(
        "tests/examples/crusI-left/Default", 2, rolling_ball_radius
    )
    example_roi = open_roi("tests/examples/crusI-left/Default/0982-1002.roi")
    area_processing(example_data, example_roi, threshold)

    # ====================================================================
    # Plot all the data
    # ====================================================================

    save_all_area_scans(
        data_dir="../data",
        out_dir="../out/area_scans",
        threshold=threshold,
        rolling_ball_radius=rolling_ball_radius,
        mouse_filter="*",
        area_filter="*lobule8*",
    )

    # ====================================================================
    # Example usage for processing multiple images and ROIs
    # ====================================================================

    results_csv = area_batch_processing(
        data_dir="../data",
        mouse_filter="*",
        area_filter="*",
        threshold=threshold,
        rolling_ball_radius=rolling_ball_radius,
    )

    save_results(
        out_directory="../out",
        file_name="results.csv",
        results=results_csv,
        metadata={"threshold": threshold, "rolling_ball_radius": rolling_ball_radius},
    )
