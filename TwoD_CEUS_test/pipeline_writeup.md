# CEUS Necrotic Area Subtraction Pipeline

## Overview

This pipeline refines a manually drawn VOI (volume of interest) on a 3D CEUS scan to exclude non-tumor tissue and necrotic regions, then performs curve quantification on the cleaned ROI. The goal is to get more accurate perfusion parameters (PE, AUC, TTP, etc.) by removing voxels that don't represent viable, well-perfused tumor tissue.

## Pipeline

```
Manual VOI + Raw 4D CEUS Scan
        │
        ▼
Step 1: Paramap generation
        (initial paramap generation)
        │
        ▼ 
Step 2: Threshold-Based VOI Segmentation
        (coarse cleanup — remove non-tumor tissue)
        │
        ▼
Step 3: Paramap Generation
        (generate parametric maps from cleaned VOI)
        │
        ▼
Step 4: Necrotic Area Subtraction
        (fine cleanup — threshold PE paramap to remove necrotic voxels)
        │
        ▼
Step 5: Curve Quantification
        (final TIC fitting on refined VOI → PE, AUC, TTP, MTT)
```

## Step 1: Threshold-Based VOI Segmentation

**Purpose:** The manually drawn VOI is intentionally oversized to ensure full tumor coverage. This step removes voxels that are clearly not part of the perfused tumor.

**Method:** Two independent criteria are applied within the VOI:

- **TTP (Time-to-Peak) threshold:** Retain voxels with TTP ≤ 75th percentile. Voxels with very late arrival times are likely outside the tumor or poorly perfused.
- **Temporal variance threshold:** Retain voxels with variance ≥ 28th percentile (top 72%). Voxels with low temporal variance show little contrast uptake/washout and are unlikely to be perfused tumor tissue.

The final mask is the intersection of both criteria (voxels must pass both). Binary hole-filling is applied to prevent interior gaps.

**Justification:** TTP and temporal variance are complementary indicators of contrast perfusion. TTP identifies tissue that enhances within a reasonable time window. Temporal variance identifies tissue that actually changes intensity over the acquisition — static tissue (no contrast uptake) is excluded. Using both reduces the chance of including non-enhancing tissue while being less aggressive than either criterion alone.

**Output:** Cleaned VOI mask (NIfTI).

## Step 3: Curve Quantification

**Purpose:** Generate per-voxel parametric maps from the cleaned VOI. These maps are needed for the next refinement step.

**Method:** For each voxel within the cleaned VOI, extract its time-intensity curve (TIC) and fit a lognormal model to derive:
- PE (Peak Enhancement)
- AUC (Area Under the Curve)
- TTP (Time to Peak)
- Other parameters as needed

**Output:** 3D parametric maps (.npy) — one per parameter.

## Step 4: Necrotic Area Subtraction

**Purpose:** Even after the coarse cleanup in Step 1, some voxels within the tumor boundary may represent necrotic (non-viable) tissue. Necrotic regions have poor contrast uptake and low PE values. This step removes them.

**Method:** Threshold the PE parametric map within the ROI:
- Compute the 15th percentile of PE values across all valid voxels in the ROI
- Remove voxels with PE below this threshold

**Justification:** PE directly measures the maximum contrast enhancement a voxel achieves. Necrotic tissue, by definition, has poor or absent perfusion and therefore low PE. A percentile-based threshold adapts to each scan's dynamic range rather than requiring an absolute cutoff. The 15th percentile was chosen as a conservative cutoff — it removes only the lowest-perfused voxels without being overly aggressive.

**Output:** Refined binary mask (NIfTI + .npy) with necrotic regions removed.

## Step 5: Curve Quantification

**Purpose:** Run final TIC analysis using the refined VOI to obtain the definitive perfusion parameters.

**Method:** Same curve quantification pipeline as Step 2, but using the necrotic-subtracted mask as the VOI. The aggregate TIC is extracted from the raw 4D scan, fit with a lognormal model, and quantitative parameters are derived.

**Output:** Final PE, AUC, TTP, MTT values representing only viable, well-perfused tumor tissue.

## Key Parameters

| Step | Parameter | Value | Rationale |
|------|-----------|-------|-----------|
| 1 | TTP percentile | 75th | Include earliest 75% of voxels by arrival time |
| 1 | Variance percentile | 72nd (top 72%) | Include most variable voxels (contrast uptake present) |
| 3 | PE percentile | 15th | Remove bottom 15% of PE values (poorest perfusion) |

## Notes

- All thresholds are percentile-based, making them adaptive to each individual scan's intensity range.
- The two-pass approach (coarse → paramaps → fine) is necessary because you need paramaps to identify necrotic tissue, but you need a reasonable VOI to generate meaningful paramaps.
- The final output is a binary mask, not a paramap — it is used as a standard VOI for curve quantification.
