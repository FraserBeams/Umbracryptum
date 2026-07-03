#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 100%-stacked bar chart for vesicle diameters by category.

Usage:
    python plot_diameter_stacked.py --input vesicle_analysis.csv --output diameter_stacked.svg

Options:
    --input   Path to input CSV file (must contain 'diameter' and 'class' columns)
    --output  Output SVG file path (default: diameter_stacked.svg)
    --help    Show this help message and exit
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

# --- Parse arguments ---
parser = argparse.ArgumentParser(
    description="Plot several SVG charts (stacked, grouped, histogram, violin) for vesicle diameters."
)
parser.add_argument("--input", "-i", required=True, help="Input CSV file (must contain 'diameter' and 'class')")
parser.add_argument("--outdir", "-o", default="plots", help="Output directory for SVGs")
args = parser.parse_args()

csv_path = args.input
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

# --- Matplotlib settings for Illustrator ---
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans']

# --- Load data ---
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    sys.exit(f" Error reading CSV file '{csv_path}': {e}")

if "diameter" not in df.columns or "class" not in df.columns:
    sys.exit(" CSV must contain columns named 'diameter' and 'class'.")

# --- Define bins ---
bins = [0, 60, 75, 90, 105, 120, 135, np.inf]
labels = ["<60", "60-75", "75-90", "90-105", "105-120", "120-135", ">135"]
df["bin"] = pd.cut(df["diameter"], bins=bins, labels=labels, right=False)

classes = ["empty", "outer_with_inner", "inner"]

# --- Grouped counts and percentages ---
grouped = df.groupby(["class", "bin"]).size().unstack(fill_value=0).reindex(classes)
percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
summary = grouped.copy()
summary["total_n"] = grouped.sum(axis=1)

print("\n=== Counts per bin ===")
print(summary)
print("\n=== Percentages per bin ===")
print(percentages.round(2))
print("\nTotal vesicles per class:")
print(grouped.sum(axis=1))

# ============================================================================
# 1. 100%-STACKED BAR PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 5))
bottom = np.zeros(len(percentages))
for label in labels:
    vals = percentages[label]
    ax.bar(percentages.index, vals, bottom=bottom, label=label)
    bottom += vals

for i, cls in enumerate(percentages.index):
    n = summary.loc[cls, "total_n"]
    ax.text(i, 102, f"n={n}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Class")
ax.set_title("Diameter distribution (100% stacked)")
ax.legend(title="Diameter (nm)", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.set_ylim(0, 110)
plt.tight_layout()
stacked_svg = os.path.join(outdir, "diameter_stacked.svg")
plt.savefig(stacked_svg, format="svg", dpi=300)
plt.close()

# ============================================================================
# 2. GROUPED BAR (per bin, category side-by-side)
# ============================================================================
# Compute percentages per bin (column normalized)
bin_sums = grouped.sum(axis=0)  # total per bin across all classes
bin_perc = grouped.copy()

for b in labels:
    total = bin_sums.get(b, 0)
    if total > 0:
        bin_perc[b] = (grouped[b] / total) * 100
    else:
        bin_perc[b] = 0

# Plot grouped bars
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(labels))
width = 0.25

for i, cls in enumerate(classes):
    ax.bar(x + (i - 1)*width, bin_perc.loc[cls], width=width, label=cls)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Diameter bin (nm)")
ax.set_ylabel("Percentage within bin (%)")
ax.set_title("Composition of classes within each diameter bin")
ax.legend(title="Class")
plt.tight_layout()
grouped_svg = os.path.join(outdir, "diameter_grouped.svg")
plt.savefig(grouped_svg, format="svg", dpi=300)
plt.close()


# ============================================================================
# 3. HISTOGRAM / KDE OVERLAY
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(data=df, x="diameter", hue="class", bins=bins, stat="percent",
             common_norm=False, multiple="dodge", edgecolor="black", ax=ax)
ax.set_xlabel("Diameter (nm)")
ax.set_ylabel("Percentage of vesicles (%)")
ax.set_title("Size distribution by class")
plt.tight_layout()
hist_svg = os.path.join(outdir, "diameter_histogram.svg")
plt.savefig(hist_svg, format="svg", dpi=300)
plt.close()

# ============================================================================
# 4. VIOLIN PLOT / BOX PLOT
# ============================================================================

def set_axes_size(ax, w, h):
    """Resize figure so that a single Axes 'ax' has exactly w×h inches of data area."""
    fig = ax.figure
    fig.canvas.draw()  # needed to get correct bbox
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    current_w, current_h = bbox.width, bbox.height
    scale_x, scale_y = w / current_w, h / current_h
    fig.set_size_inches(fig.get_size_inches()[0] * scale_x,
                        fig.get_size_inches()[1] * scale_y)
    fig.tight_layout(pad=0.1)
    return fig

fig, ax = plt.subplots()
sns.violinplot(data=df, x="class", y="diameter", inner="box", palette="Set2", ax=ax)
ax.set_xlabel("Class", fontsize=8)
ax.set_ylabel("Diameter (nm)", fontsize=8)
ax.tick_params(axis="both", labelsize=8)
plt.tight_layout()
violin_svg = os.path.join(outdir, "diameter_violin.svg")
set_axes_size(ax, 2, 2)
plt.savefig(violin_svg, format="svg", bbox_inches="tight", dpi=300)
plt.close()

# ============================================================================
print("\n Saved all SVG plots to:")
for f in [stacked_svg, grouped_svg, hist_svg, violin_svg]:
    print("  ", os.path.abspath(f))

