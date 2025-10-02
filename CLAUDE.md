# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based fluorescent microscopy image analysis tool for quantifying fluorescence in cerebellar regions. It processes multi-channel TIF images from ImageJ with ROI files to perform two types of analysis:

1. **Area Analysis**: Measures the percentage of fluorescent pixels above a threshold within ROIs
2. **Particles Analysis**: Counts individual cells using DAPI staining (channel 1) and determines which cells express IEG markers (channel 2) using watershed segmentation

## Commands

### Testing
```bash
pytest                                    # Run all tests
pytest tests/area_analysis_test.py        # Run specific test file
pytest -k test_name                       # Run specific test
```

### Running Analysis

Both analysis types are executed via their respective scripts with parameters configured directly in the code:

```bash
python src/run_area_analysis.py           # Area-based fluorescence analysis
python src/run_particles_analysis.py      # Cell-counting particle analysis
```

### Plotting Results

```bash
python src/plots.py --csv path/to/results.csv                    # Interactive plot (Plotly)
python src/plots.py --csv path/to/results.csv --static           # Static plot (matplotlib)
python src/plots.py --csv-dir path/to/csv/directory              # Process multiple CSVs
```

## Architecture

### Data Flow

1. **Input**: TIF images (multi-channel) + ROI files (ImageJ format) organized as `data_dir/mouse_dir/area_dir/Default/`
2. **Processing**: Apply rolling ball background subtraction (optional, cached), threshold, and ROI masking
3. **Analysis**: Area percentage (area_analysis.py) or particle counting with watershed segmentation (particles_analysis.py)
4. **Output**: CSV files with metadata headers (commented lines starting with `#`)

### Key Modules

- **commun.py**: Core utilities shared across analysis types
  - `batch_iterator()`: Main iterator for processing multiple experiments; handles file discovery, caching, and error recovery
  - `open_image()`: Loads TIF images with optional rolling ball background subtraction (cached as `.npy` files to avoid slow reprocessing)
  - `open_roi()`: Loads ImageJ ROI files using roifile library
  - `save_results()`: Writes CSV files with metadata in commented header lines
  - Image channels: 1=DAPI (nuclei), 2=IEG (immediate early gene markers)

- **area_analysis.py**: Area-based fluorescence quantification
  - `area_batch_processing()`: Multi-threaded batch processing (n_jobs=-2 uses all cores minus 2)
  - `area_processing()`: Single-image processing; calculates ratio of threshold pixels within ROI
  - Uses channel 2 (IEG) only

- **particles_analysis.py**: Cell counting with dual-channel analysis
  - `particles_batch_processing()`: Sequential processing of all experiments
  - `particles_processing_dapi()`: Watershed segmentation on DAPI channel to identify individual cells; returns cell count and labels
  - `particles_processing_ieg()`: Checks which DAPI-labeled cells have IEG expression
  - Processing: Gaussian filter → threshold → distance transform → watershed with percentile-based markers → small object removal

- **plots.py**: Visualization and statistical plotting
  - Two plot types: static (matplotlib/seaborn) and interactive (Plotly)
  - `plot_data()` / `plot_data_interactive()`: Visualize single analysis results with 4-panel layout (image, ROI, threshold mask, combined mask/particles)
  - `plot_results()` / `plot_results_interactive()`: Generate grouped bar plots with scatter overlay; auto-detects plot type from CSV columns (roi_rate vs particles_rate)
  - CSV parsing: Handles metadata in commented header lines (`# key: value`), infers experimental groups from mouse names, aggregates multiple slices per region

### Important Processing Details

- **Caching**: Rolling ball background subtraction is slow, so processed images are cached as `.npy` files in the same directory as the original TIF. Use `use_cache=False` to disable.
- **Channel numbering**: Files are named `img_channel001_*.tif` (DAPI) and `img_channel002_*.tif` (IEG)
- **Batch processing**: `batch_iterator()` expects exactly one TIF per channel and one ROI file per experiment directory. It logs warnings and skips directories with incorrect file counts.
- **Parallelization**: Area analysis uses joblib threading (`n_jobs=-2`), while particle analysis is sequential due to higher memory requirements
- **CSV metadata**: All CSVs include processing parameters as commented lines at the top for reproducibility
- **Slice aggregation**: Plots automatically group multiple slices (e.g., `area_slice1`, `area_slice2`) by extracting and averaging slice numbers

### Test Data Structure

Test data is in `examples/` and `tests/examples/` with the expected structure:
```
mouse_A/
  area_slice1_20x/Default/
    img_channel002_*.tif
    *.roi
  area_slice2_20x/Default/
    ...
```

Tests validate both single-image processing and batch processing with specific expected values (e.g., `test_simple_single_processing` expects roi_rate ≈ 0.44396).
