#!/usr/bin/env python3
import re
import argparse
import pandas as pd

# 1) List of regions to average over:
REGIONS = [
    'simplex', 'crusI', 'crusII', 'paramedian',
    'paraflocculus', 'lobule2', 'lobule3', 'lobule45',
    'lobule6', 'lobule7', 'lobule8', 'lobule9', 'lobule10'
]

# Build a case‐insensitive regex that will find any of the above in the area_name
REGION_PATTERN = re.compile(r'(' + '|'.join(REGIONS) + r')', re.IGNORECASE)
# Map lowercase→canonical
REGION_MAP = {r.lower(): r for r in REGIONS}

def extract_region(name: str) -> str:
    """
    Return the canonical region name found in `name`, or None if none match.
    """
    m = REGION_PATTERN.search(name)
    if not m:
        return None
    found = m.group(1).lower()
    return REGION_MAP[found]

def average_regions(input_csv: str, output_csv: str):
    # 2) Load your data
    df = pd.read_csv(input_csv, comment='#')

    # 3) Find which column contains your measurement
    #    (we look for 'roi_rate' or 'mean_roi_rate')
    for col in ('roi_rate', 'mean_roi_rate'):
        if col in df.columns:
            value_col = col
            break
    else:
        raise KeyError("Couldn't find 'roi_rate' or 'mean_roi_rate' in your CSV columns: "
                       + ", ".join(df.columns))

    # 4) Extract the region from each row
    df['region'] = df['area_name'].astype(str).apply(extract_region)
    df = df.dropna(subset=['region'])

    # 5) Group and compute mean
    out = (
        df
        .groupby(['mouse_name', 'region'], as_index=False)[value_col]
        .mean()
        .rename(columns={'region': 'area_name', value_col: 'roi_rate'})
    )

    # 6) Save
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} rows to {output_csv}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Average ROI measurements by mouse and cerebellar region."
    )
    p.add_argument(
        '--input', '-i', required=True,
        help="Path to input CSV file"
    )
    p.add_argument(
        '--output', '-o', default='averaged_regions.csv',
        help="Path to write the averaged CSV"
    )
    args = p.parse_args()
    average_regions(args.input, args.output)
