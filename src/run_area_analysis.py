from commun import open_image, open_roi, save_results
from particles_analysis import (
    particles_batch_processing,
    save_all_debug_grids,
)

if __name__ == "__main__":
    # Meta-parameters
    dapi_threshold = 900
    ieg_threshold = 150
    gaussian_sigma = 1.5
    min_particle_size = 30
    markers_percentile = 90
    rolling_ball_radius = (
        15  # Set to None to disable rolling ball background subtraction
    )

    # --- ImageJ-style circularity filter (range 0.0–1.0) ---
    # Typical choices: 0.70–1.00 (roundish), 0.85–1.00 (very round).
    # To disable filtering, use 0.0–1.0.
    min_circularity = 0.70
    max_circularity = 1.00

    # ====================================================================
    # Example usage for one image and one ROI
    # ====================================================================

 example_data = open_image(
        "data/L_CrusI_20x_center_left/Default", 1, rolling_ball_radius
    )
    example_roi = open_roi("data/L_CrusI_20x_center_left/Default/1006-0970.roi")
    particles_processing_threshold(
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

    save_all_debug_grids(
        data_dir="F:/Complete dataset for project 1/MLI particle analysis/Additional data Nov 2025",
        out_dir="../out/particle_debug_grids/extra260",
        dapi_threshold=dapi_threshold,
        ieg_threshold=ieg_threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        rolling_ball_radius=rolling_ball_radius,
        mouse_filter="*",
        area_filter="*",
        min_circularity=min_circularity,
        max_circularity=max_circularity,
        clip_percentiles=(1, 99),  # widen contrast
        gamma=0.9,  # lighten midtones (<1 brightens)
        gain=1.3,  # final brightness bump
    )

    # ====================================================================
    # Example usage for processing multiple images and ROIs
    # ====================================================================

    results_csv = area_batch_processing(
        data_dir="../data",
        mouse_filter="*",
        area_filter="*",
        dapi_threshold=dapi_threshold,
        ieg_threshold=ieg_threshold,
        gaussian_sigma=gaussian_sigma,
        min_particle_size=min_particle_size,
        markers_percentile=markers_percentile,
        rolling_ball_radius=rolling_ball_radius,
        min_circularity=min_circularity,
        max_circularity=max_circularity,
    )

    save_results(
        out_directory="../out",
        file_name="results.csv",
        results=results_csv,
        metadata={
            "dapi_threshold": dapi_threshold,
            "ieg_threshold": ieg_threshold,
            "gaussian_sigma": gaussian_sigma,
            "min_particle_size": min_particle_size,
            "markers_percentile": markers_percentile,
            "rolling_ball_radius": rolling_ball_radius,
            "min_circularity": min_circularity,
            "max_circularity": max_circularity,
        },
    )
