"""
Quick script to deduplicate features.csv by keeping only one entry per unique ROI.

This fixes the data leakage issue in LOOCV.
"""

import pandas as pd
import numpy as np

# Load features
df = pd.read_csv("results/biomarkers/features.csv")

print("=" * 80)
print("DEDUPLICATING FEATURES")
print("=" * 80)
print(f"\nOriginal: {len(df)} rows")

# Create ROI ID
df["roi_id"] = df["patient_id"].astype(str) + "_" + df["region"].astype(str)
print(f"Unique ROIs: {df['roi_id'].nunique()}")

# Keep first occurrence of each ROI (you could also take mean/median)
df_dedup = df.drop_duplicates(subset="roi_id", keep="first")

print(f"Deduplicated: {len(df_dedup)} rows")
print(f"\nRemoved {len(df) - len(df_dedup)} duplicate rows")

# Drop the temporary roi_id column before saving
df_dedup = df_dedup.drop(columns=["roi_id"])

# Save
output_path = "results/biomarkers/features_dedup.csv"
df_dedup.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

print("\n⚠️  WARNING: Different spectra inference runs had different ADC values!")
print("   Keeping 'first' entry - consider averaging or selecting best run instead.")
