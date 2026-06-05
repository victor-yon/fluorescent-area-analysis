# Fluorescent Area Analysis

Python tools for quantifying fluorescence in cerebellar regions from multi-channel TIF images acquired in ImageJ.

---

## Data structure

All scripts expect data organised as follows:

```
data_dir/
  mouse_name/
    area_name/
      Default/
        img_channel001_*.tif   ← DAPI (nuclei)
        img_channel002_*.tif   ← IEG marker
        *.roi                  ← ImageJ ROI file
```

Each `Default/` folder must contain exactly one TIF per channel and exactly one `.roi` file. Directories that do not meet this requirement are skipped with a warning.

---

## Analysis scripts

Both scripts are configured by editing parameters directly at the top of the `if __name__ == "__main__":` block, then run from the `src/` directory.

### Area analysis — `run_area_analysis.py`

Measures the **percentage of pixels above a threshold** within each ROI, using only the IEG channel (channel 2).

```bash
python src/run_area_analysis.py
```

**Key parameters to set in the script:**

| Parameter | Description |
|---|---|
| `data_dir` | Path to the root data folder |
| `ieg_threshold` | Pixel intensity threshold for fluorescence detection |
| `rolling_ball_radius` | Radius for background subtraction (set `None` to disable) |
| `mouse_filter` | Glob pattern to select mouse subfolders (e.g. `"NC*"`, `"*"` for all) |
| `area_filter` | Glob pattern to select area subfolders |

**Output CSV columns:** `mouse_name`, `area_name`, `roi_area_px`, `roi_rate`

`roi_rate` is the fraction (0–1) of ROI pixels above the threshold. Multiply by 100 for a percentage.

---

### Particles analysis — `run_particles_analysis.py`

Counts individual DAPI-stained cell nuclei using watershed segmentation, then determines what fraction are IEG-positive.

```bash
python src/run_particles_analysis.py
```

**Key parameters to set in the script:**

| Parameter | Description |
|---|---|
| `data_dir` | Path to the root data folder |
| `dapi_threshold` | Intensity threshold for nucleus detection (channel 1) |
| `ieg_threshold` | Intensity threshold for IEG positivity (channel 2) |
| `gaussian_sigma` | Blur applied before thresholding (reduces noise) |
| `min_particle_size` | Minimum nucleus area in pixels (removes debris) |
| `markers_percentile` | Percentile used to seed watershed (higher = fewer seeds) |
| `rolling_ball_radius` | Background subtraction radius (`None` to disable) |
| `min_circularity` / `max_circularity` | Shape filter (0–1); `0.70`–`1.00` keeps round nuclei |

**Output CSV columns:** `mouse_name`, `area_name`, `roi_area_px`, `nb_particles_dapi`, `nb_particles_ieg_overlapping_surface`, `nb_particles_ieg_threshold_watershed`, `particles_rate`

`particles_rate` is the primary result: fraction of DAPI nuclei that are IEG-positive (`nb_particles_ieg_overlapping_surface / nb_particles_dapi`).

**Debug grid output**

The script also exposes `save_all_debug_grids()`, which saves a 9-panel PNG per sample showing the raw images, preprocessed masks, and detected particles at each stage. Configure `out_dir` and the display parameters (`clip_percentiles`, `gamma`, `gain`) in the script to use it.

---

### Caching

Rolling ball background subtraction is slow. Processed images are automatically cached as `.npy` files alongside the original TIFs and reused on subsequent runs. To force reprocessing, set `use_cache=False` or delete the `.npy` files.

---

## Running tests

```bash
pytest                              # all tests
pytest tests/area_analysis_test.py  # one file
pytest -k test_name                 # one test
```

Test data is in `tests/examples/` with the same folder structure as production data.
