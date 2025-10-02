from src.commun import open_image, open_roi, save_results
from src.particles_analysis import (
    particles_batch_processing,
    particles_processing_dapi,
    save_all_particle_scans,
)

if __name__ == "__main__":
    # Meta-parameters
    dapi_threshold = 1624
    ieg_threshold = 800
    gaussian_sigma = 2
    min_particle_size = 30
    markers_percentile = 90
    rolling_ball_radius = (
        90  # Set to None to disable rolling ball background subtraction
    )

    # ====================================================================
    # Example usage for one image and one ROI
    # ====================================================================

    example_data = open_image(
        "data/L_CrusI_20x_center_left/Default", 1, rolling_ball_radius
    )
    example_roi = open_roi("data/L_CrusI_20x_center_left/Default/1006-0970.roi")
    particles_processing_dapi(
        example_data,
        example_roi,
        ieg_threshold,
        gaussian_sigma,
        min_particle_size,
        markers_percentile,
    )

    # ====================================================================
    # Plot all the data
    # ====================================================================

    save_all_particle_scans(
        data_dir="../data",
        out_dir="../out/particle_scans",
        threshold=ieg_threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        rolling_ball_radius=rolling_ball_radius,
        mouse_filter="*",
        area_filter="*lobule8*",
    )

    # ====================================================================
    # Example usage for processing multiple images and ROIs
    # ====================================================================

    results_csv = particles_batch_processing(
        data_dir="../data",
        mouse_filter="*",
        area_filter="*",
        dapi_threshold=dapi_threshold,
        ieg_threshold=ieg_threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        rolling_ball_radius=rolling_ball_radius,
    )

    save_results(
        out_directory="../out",
        file_name="particles_results.csv",
        results=results_csv,
        metadata={
            "dapi_threshold": dapi_threshold,
            "ieg_threshold": ieg_threshold,
            "gaussian_sigma": gaussian_sigma,
            "min_particle_size": min_particle_size,
            "markers_percentile": markers_percentile,
            "rolling_ball_radius": rolling_ball_radius,
        },
    )
