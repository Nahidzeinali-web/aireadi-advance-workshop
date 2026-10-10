# Sample data for the workshop notebook

A small slice of the **Synthetic AI-READI Dataset for T2DM Research** made by the ADVANCE Center at Vanderbilt
University Medical Center ([doi:10.5281/zenodo.22944362](https://doi.org/10.5281/zenodo.22944362)). Everything here
is synthetic: computer-generated to resemble the real AI-READI study, with no real person behind any record.

## Use of this data is restricted

This folder is covered by the **AI-READI Synthetic Data License Agreement, Version 1.0** (Licensor: Washington
University in St. Louis), included here as `LICENSE-AI-READI-synthetic-data.pdf`. Every copy, including these PNG
conversions and the trimmed table, stays under that agreement. In short:

* Only people covered by the license may use it. A license accepted by an individual covers that person only; a
  license accepted for a group covers members of that group, and each of them must receive a copy of the agreement.
* Do not post or share it as a standalone download (public repository, zip file, code package) without written
  consent from the Licensor.
* Do not try to identify, locate or contact any real person behind the source data, and do not use it for
  clinical decisions.
* Figures and tables made from it may appear in papers and presentations, with the acknowledgement below.

## What is here

| Path | Contents |
|---|---|
| `scans/scan_0001.png` ... `scan_2000.png` | retina cross-section scans (optical coherence tomography), 256 x 256 grayscale, 500 from each of four machines: Zeiss Cirrus, Topcon Maestro2, Heidelberg Spectralis, Topcon Triton |
| `scans.csv` | for each scan: machine, maker, part of the eye, left or right eye, original file name |
| `gallery/*.png` + `gallery.csv` | one example of each kind of picture from each machine (13 images) |
| `participants.csv` | 16 measurements for 10,518 synthetic participants (one row each) |

`participants.csv` columns and where they come from:

| Column | Source in the full dataset |
|---|---|
| `type2_diabetes`, `high_blood_pressure`, `kidney_problems` | `condition_occurrence.csv`, self-reported, 1 = yes |
| `hba1c_percent`, `lab_glucose_mg_dl`, `creatinine_mg_dl` | `measurement.csv`, blood tests |
| `body_mass_index`, `waist_cm`, `systolic_blood_pressure_mmhg` | `measurement.csv`, clinic visit |
| `glucose_monitor_mean_mg_dl`, `glucose_monitor_variability_mg_dl` | `synthetic_wearable_activity_glucose.tsv`, continuous glucose monitor summary |
| `watch_heart_rate_bpm`, `watch_resting_heart_rate_bpm`, `watch_steps`, `watch_stress_score` | `synthetic_wearable_activity_glucose.tsv`, wrist-worn activity tracker summary |
| `clinic_heart_rate_bpm` | `cardiac_ecg/manifest.tsv`, heart rate measured by the clinic heart tracing |

The source tables do not state units; the units in the column names are the usual ones for each test and match
the value ranges. The watch step and stress values are summaries whose units are not documented, so treat them
as relative.

## How it was made

`tools/build_sample_data.py` builds this folder from the dataset's storage bucket with fixed random seeds, so anyone
with their own access key can rebuild the identical sample. Images were converted from DICOM to PNG without any
change to the pixel values. The original file name of every image is kept in `scans.csv` and `gallery.csv`.

## Acknowledgement

When reporting work that uses this data, acknowledge:

* the Licensor, Washington University in St. Louis;
* the synthetic dataset: Jackson N, Espinosa Dice N, Yan C, Li Z, Jiang X, Lee A, Malin B. *Synthetic AI-READI
  Dataset for T2DM Research.* Zenodo, 2026. doi:10.5281/zenodo.22944362;
* the source study, AI-READI: AI-READI Consortium, *Nature Metabolism* 2024, doi:10.1038/s42255-024-01165-x, and
  the AI-READI dataset v3.0.0, doi:10.60775/fairhub.3 (current citation at docs.aireadi.org);
* the funders: NIH Bridge2AI OT2OD032644, National Library of Medicine K99LM014428 and T15LM007450, and National
  Human Genome Research Institute U54HG012510.
