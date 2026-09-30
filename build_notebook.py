"""Build aireadi_synthetic_demo.ipynb and PRESENTER_NOTES.md from one cell list.

Each cell carries its presenter notes in metadata["presenter_notes"], so the notebook
and the notes file never drift apart. Edit this file, then run:

    python build_notebook.py
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
NB_PATH = HERE / "aireadi_synthetic_demo.ipynb"
NOTES_PATH = HERE / "PRESENTER_NOTES.md"

CELLS = []


def md(src, notes="", minutes=None, optional=False):
    CELLS.append(dict(kind="md", src=src.strip("\n"), notes=notes.strip(), minutes=minutes, optional=optional))


def code(src, notes="", minutes=None, optional=False):
    CELLS.append(dict(kind="code", src=src.strip("\n"), notes=notes.strip(), minutes=minutes, optional=optional))


# ---------------------------------------------------------------------------
# 0. Title
# ---------------------------------------------------------------------------
md(r"""
# AI-READI in the Cloud: a hands-on tour of the synthetic T2DM dataset

**Dataset:** ADVANCE Center Synthetic AI-READI Dataset for T2DM Research (read-only S3 bucket)
**Runs on:** any Jupyter environment with network access to S3. A GPU is optional.

**What we will do**

1. Meet AI-READI and its synthetic twin
2. Connect to the data in the cloud
3. Tour every data type: retinal photos, OCT, ECG, wearables and CGM, clinical records
4. Look for relationships across data types
5. Play a game: *which device took this scan?* Humans vs a small neural network
6. Leave with research questions worth a grant
""",
notes="""
Welcome everyone. Today is about how quickly you can go from zero to real questions on a large multimodal
diabetes dataset, entirely in the cloud. No deep learning background is needed.

Frame the session: everything you see runs from one notebook against a read-only S3 bucket. Nothing is
downloaded in bulk; we pull only what each step needs.

Mention that attendees can follow along on their own VM if they have credentials, or just watch.
""", minutes=1)

# ---------------------------------------------------------------------------
# 1. Intro to AI-READI
# ---------------------------------------------------------------------------
md(r"""
## 1. What is AI-READI?

**AI-READI** (Artificial Intelligence Ready and Exploratory Atlas for Diabetes Insights) is one of the flagship
data generation projects of the NIH Common Fund **Bridge2AI** program. Its aim is a dataset built for AI from day
one, to study **salutogenesis** in type 2 diabetes: what separates people who move toward health from those who
do not.

**Who is in it.** The study targets 4,000 adults aged 40 and older, recruited at UAB, UCSD and UW, and balanced by
design across three factors:

| Factor | Groups |
|---|---|
| Diabetes severity | no diabetes; prediabetes or lifestyle-controlled; oral medication or non-insulin injectables; insulin |
| Race and ethnicity | Asian, Black, Hispanic, White |
| Sex | female, male |

**What is measured.** Each participant contributes many data types, each stored in a community standard:

| Data type | Devices | Standard |
|---|---|---|
| Retinal photography (color, infrared, autofluorescence) | Optomed Aurora, iCare Eidon, Topcon Maestro2, Topcon Triton, Heidelberg Spectralis, Zeiss Cirrus | DICOM |
| Retinal OCT and OCT angiography | Heidelberg Spectralis, Topcon Maestro2, Topcon Triton, Zeiss Cirrus | DICOM |
| 12-lead ECG | Philips PageWriter TC30 | WFDB |
| Continuous glucose monitoring (10 days) | Dexcom G6 | Open mHealth JSON |
| Physical activity, heart rate, sleep, stress | Garmin Vivosmart 5 | Open mHealth JSON |
| Home environment (particulates, VOCs, light) | LeeLab Anura sensor | CSV |
| Clinical data, labs, surveys, cognition | study visit | OMOP CDM |

**Releases.** v1.0.0 (May 2024, 204 pilot participants), v2.0.0 (Nov 2024, 863 added), v3.0.0 (Nov 2025, 1,213
added, about 2,280 in total and 3.82 TB). The real data is distributed through **FAIRhub** under a data license
agreement; a few sensitive fields need Data Access Committee approval.

Sources: [docs.aireadi.org](https://docs.aireadi.org/), [changelog](https://docs.aireadi.org/docs/3/changelog),
[protocol paper, BMJ Open 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800295/),
[Nature Metabolism 2024](https://doi.org/10.1038/s42255-024-01165-x), [aireadi.org](https://aireadi.org/).
""",
notes="""
Two or three minutes, no more. The points that matter for the rest of the demo:

1. It is deliberately balanced by diabetes severity and by race and ethnicity. That design is what makes
   fairness and "path back to health" questions answerable.
2. It is multimodal per person: eyes, heart, glucose, activity, environment, labs. Most diabetes datasets have
   one or two of these.
3. The retinal images come from six different devices. Remember this; it is the setup for the game in Part 5.
4. Everything is in a standard format (DICOM, WFDB, OMOP), so the same code works on the real release.

If asked about access: real data is on FAIRhub (fairhub.io) under a license; restricted fields such as
race/ethnicity, sex, zip code and medications need a separate request. Use is limited to T2DM research.
""", minutes=3)

md(r"""
### 1.1 Why a synthetic version?

The real release needs a data license, and some fields need committee approval. To let people explore, teach and
prototype first, the **ADVANCE Center** at Vanderbilt University Medical Center built a synthetic companion from
AI-READI v3.0.0, published on the AWS Marketplace as *ADVANCE Center Synthetic AI-READI Dataset for T2DM Research*
and archived on [Zenodo](https://doi.org/10.5281/zenodo.22944362).

How it was made ([code](https://github.com/nicholas-j-jackson/ai_readi_synth)):

* **Images:** a latent diffusion model trained on 256 x 256 images and conditioned on each image's
  device, anatomical region, laterality (left or right eye) and disease labels.
* **Tabular data:** a reinforcement-learning based generator that produces whole participant records.
* The three modules (fundus photos, OCT, tabular) were **generated independently**.

That last point shapes what we can and cannot ask:

| Module | Records | Linked to |
|---|---|---|
| Retinal photography | 93,856 images | its own AMD, glaucoma and diabetic retinopathy labels |
| Retinal OCT | 56,512 B-scans | its own AMD, glaucoma and diabetic retinopathy labels |
| Tabular cohort | 10,518 synthetic participants | clinical labs, conditions, ECG measurements, wearable and CGM summaries, all sharing one `participant_id` |

Images are **not** linked to the tabular participants. A `participant_id` of 42 in the OCT folder and 42 in the ECG
folder are unrelated synthetic people.
""",
notes="""
This is the most important caveat of the talk, so say it plainly: synthetic data is for exploring, teaching and
building pipelines. Findings have to be confirmed on the real data.

Explain the linkage table. Inside the tabular cohort, the generator produced whole participants, so labs, ECG,
wearable and CGM values for participant 42 belong together and their relationships were learned from real data.
Images were generated separately, one synthetic person per image, so you cannot link a retina to an HbA1c here.

Why ADVANCE: it is a Vanderbilt center (Biomedical Informatics) with deep privacy expertise; Brad Malin is a
co-director and an author on the synthetic dataset, together with Aaron Lee.
""", minutes=2)

# ---------------------------------------------------------------------------
# 2. Connect
# ---------------------------------------------------------------------------
md(r"""
## 2. Connect to the data

The bucket is read-only and accessed with **temporary AWS credentials**. The instructions say they last 12 hours; in our testing they expired after about **one hour**, so download a fresh set right before you start. You get them as a ZIP
containing `credentials.csv`. The notebook looks for credentials in this order:

1. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` already set in the environment
2. a `credentials.csv` file, at the path in `AIREADI_CREDS_CSV` or in the working directory

Never paste keys into a notebook cell; notebooks get shared and committed.
""",
notes="""
While this cell is on screen, show where the credentials ZIP lives on the VM (or say you exported the three
environment variables before starting Jupyter). Point out the expiry: the instructions promise 12 hours but in our
testing the tokens died after about one hour. Download the ZIP within the hour before the session. If a cell
starts failing with ExpiredToken, download a fresh ZIP and rerun the credentials cell (the cache means nothing
already downloaded is lost).
""", minutes=1)

code(r"""
# Install anything a fresh VM image is missing (quiet, one time).
import importlib.util, subprocess, sys
NEEDED = {"boto3": "boto3", "pydicom": "pydicom", "wfdb": "wfdb", "seaborn": "seaborn",
          "sklearn": "scikit-learn", "torch": "torch"}
missing = [pkg for mod, pkg in NEEDED.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("installed:", missing or "nothing, all present")
""",
notes="""
Run it and move on. On a Deep Learning VM image everything is usually present already. If torch has to be
installed live it can take a few minutes, so do this before the session starts.
""", minutes=0.5)

code(r"""
import os, csv, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3, botocore, pydicom, wfdb

BUCKET = os.environ.get("AIREADI_BUCKET", "ai-readi-sd")     # identical mirror: "ai-readi-bucket" in us-east-1
REGION = os.environ.get("AIREADI_REGION", "us-east-2")
ROOT = "ai-readi"                                            # every key starts with this prefix
CACHE = Path(os.environ.get("AIREADI_CACHE", "~/aireadi_cache")).expanduser()
SEED = 42

# One chart style for the whole notebook: light surface, quiet axes, thin marks.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 10, "lines.linewidth": 2, "figure.dpi": 110, "axes.axisbelow": True,
})
pd.set_option("display.max_columns", 30)
print("bucket:", BUCKET, "| region:", REGION)
""",
notes="""
Configuration lives in one place. Mention the two identical buckets: ai-readi-sd in us-east-2 and
ai-readi-bucket in us-east-1. Pick the one in the same region as your VM; transfers inside a region are fast
and free.

The cache folder means every file is downloaded once. Rerunning the notebook is quick.
""", minutes=0.5)

code(r"""
def load_credentials(csv_path=os.environ.get("AIREADI_CREDS_CSV", "credentials.csv")):
    # Use AWS_* variables if they are set, otherwise read the credentials.csv issued with the dataset.
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return "environment variables"
    row = next(csv.DictReader(open(Path(csv_path).expanduser())))
    os.environ["AWS_ACCESS_KEY_ID"] = row["access_key"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = row["secret_key"]
    os.environ["AWS_SESSION_TOKEN"] = row["session_token"]
    return str(csv_path)

source = load_credentials()
role = boto3.client("sts").get_caller_identity()["Arn"].split("/")[-2]
s3 = boto3.client("s3", region_name=REGION,
                  config=botocore.config.Config(max_pool_connections=64))
print(f"credentials from {source}; assumed role: {role}")
""",
notes="""
The output should say the role is s3-readonly. Point out that the notebook never prints the keys.

If this fails: ExpiredToken means the token has lapsed (about an hour in practice); AccessDenied usually means the variables from an older
ZIP are still exported in the shell. Restart the kernel after exporting new ones.
""", minutes=0.5)

md(r"""
### 2.1 What is in the bucket

Listing all 198,124 objects takes a minute, so here is the inventory up front. The cell below walks the top two
levels live.

| Prefix | Contents | Files | Size |
|---|---|---:|---:|
| `fundus/retinal_photography/` | color (CFP), infrared (IR) and autofluorescence (FAF) photos, DICOM, plus `manifest.tsv` | 93,861 | 18.6 GB |
| `fundus/clinical_data/` | `observation.csv`: AMD, glaucoma, diabetic retinopathy per image (OMOP) | 1 | 52 MB |
| `oct/retinal_oct/` | structural OCT B-scans, DICOM, plus `manifest.tsv` | 83,221 | 10.5 GB |
| `oct/clinical_data/` | `observation.csv`: AMD, glaucoma, diabetic retinopathy per B-scan (OMOP) | 1 | 31 MB |
| `tabular/cardiac_ecg/` | one 12-lead ECG record per participant (WFDB `.hea` + `.dat`; waveforms are flat placeholders, measurements are synthetic), plus `manifest.tsv` | 21,037 | 1.4 GB |
| `tabular/clinical_data/` | `measurement.csv` (73 vitals, labs and cognitive scores), `condition_occurrence.csv` (30 conditions) | 2 | 131 MB |
| `tabular/synthetic_wearable_activity_glucose.tsv` | wearable and CGM summary features, one row per participant | 1 | 2 MB |

In total 30.7 GB. The OCT folder holds 26,708 extra 512 x 512 scans that are not in the manifest; this notebook uses
the manifest, which lists the 256 x 256 B-scans.
""",
notes="""
Walk the table top to bottom. Stress that the manifests and clinical CSVs are tiny (well under 300 MB together),
so you can explore metadata for the whole dataset before downloading a single image.

The note about 26,708 unlisted OCT files is a real quirk we found; mention it only if someone asks about file
counts.
""", minutes=1)

code(r"""
def ls(prefix):
    # Immediate children of a prefix, like `aws s3 ls`.
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"] for p in r.get("CommonPrefixes", [])] + [o["Key"] for o in r.get("Contents", [])]

for top in ls(f"{ROOT}/"):
    print(top)
    for child in ls(top)[:6]:
        print("   ", child.removeprefix(top))
""",
notes="""
Live proof that we are reading from S3. Every data type has a clinical_data folder or a manifest next to the
files. That pattern mirrors the real AI-READI release.
""", minutes=0.5)

code(r"""
def fetch(key):
    # Download s3://BUCKET/key into the local cache once and return the local path.
    dst = CACHE / key
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(dst))
    return dst

def fetch_many(keys, workers=32):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fetch, keys))

SMALL_FILES = [
    "fundus/retinal_photography/manifest.tsv", "fundus/clinical_data/observation.csv",
    "oct/retinal_oct/manifest.tsv", "oct/clinical_data/observation.csv",
    "tabular/cardiac_ecg/manifest.tsv", "tabular/synthetic_wearable_activity_glucose.tsv",
    "tabular/clinical_data/measurement.csv", "tabular/clinical_data/condition_occurrence.csv",
]
t0 = time.time()
paths = dict(zip(SMALL_FILES, fetch_many([f"{ROOT}/{k}" for k in SMALL_FILES])))
size_mb = sum(p.stat().st_size for p in paths.values()) / 1e6
print(f"{len(paths)} metadata and tabular files, {size_mb:.0f} MB, {time.time() - t0:.0f} s")
""",
notes="""
Two helpers carry the whole notebook: fetch one key into the cache, or fetch many in parallel threads.
The parallel version matters for images: S3 is fast per connection but has per-request latency, so 32 threads
is roughly 20 to 30 times faster than a plain loop.

Expect around 10 to 30 seconds in-region for about 250 MB.
""", minutes=0.5)

# ---------------------------------------------------------------------------
# 3. Tour
# ---------------------------------------------------------------------------
md(r"""
## 3. A tour of the data types

### 3.1 Retinal images: what was captured, on which device
""",
notes="""
Transition: we have the metadata for all 150,000 images on disk now. First question is always what was captured
and on which machine.
""", minutes=0.25)

code(r"""
fundus = pd.read_csv(paths["fundus/retinal_photography/manifest.tsv"], sep="\t", index_col=0)
octs = pd.read_csv(paths["oct/retinal_oct/manifest.tsv"], sep="\t", index_col=0)
fundus["key"] = f"{ROOT}/fundus/" + fundus.filepath
octs["key"] = f"{ROOT}/oct/" + octs.filepath
octs["imaging"] = "OCT"
images = pd.concat([fundus, octs], ignore_index=True)
images["device"] = images.manufacturers_model_name

IMAGING = ["Color Photography", "Infrared Reflectance", "Autofluorescence", "OCT"]
DEVICE_ORDER = ["Aurora", "Eidon", "Maestro2", "Triton", "Spectralis", "Cirrus"]
counts = pd.crosstab(images.imaging, images.device).reindex(index=IMAGING, columns=DEVICE_ORDER)
counts.fillna(0).astype(int).map(lambda v: f"{v:,}" if v else "")
""",
notes="""
This matrix is the design of the imaging arm: the same eye photographed by many devices. Point out that color
photos come from four devices, infrared from four, OCT from four, and only the Eidon does autofluorescence.
Blank cells mean that device does not capture that type.

For the real data this is what makes cross-device research possible: the same people, the same day, different
machines.
""", minutes=1)

code(r"""
def read_pixels(path):
    return pydicom.dcmread(path).pixel_array

gallery = images.groupby(["imaging", "device"]).sample(1, random_state=SEED)
gallery["local"] = fetch_many(gallery.key)

fig, axes = plt.subplots(len(IMAGING), len(DEVICE_ORDER), figsize=(13, 9))
for i, imaging in enumerate(IMAGING):
    for j, device in enumerate(DEVICE_ORDER):
        ax = axes[i, j]
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        row = gallery[(gallery.imaging == imaging) & (gallery.device == device)]
        if row.empty:
            ax.text(0.5, 0.5, "not captured", ha="center", va="center", color=MUTED, fontsize=9)
        else:
            px = read_pixels(row.local.iloc[0])
            if imaging in ("Infrared Reflectance", "Autofluorescence") and px.ndim == 3:
                px = px.mean(axis=2)
            ax.imshow(px, cmap=None if px.ndim == 3 else "gray")
        if i == 0:
            ax.set_title(device)
        if j == 0:
            ax.set_ylabel(imaging.replace(" ", "\n"), rotation=0, ha="right", va="center", color=INK2)
fig.suptitle("One synthetic image per imaging type and device", x=0.01, ha="left", fontweight="bold")
plt.tight_layout()
""",
notes="""
Give the audience ten seconds to look. These are all synthetic, generated by a diffusion model, yet they carry
each device's look: the Aurora's small logo in the corner, the Eidon's text overlays and mosaic shape, the
Triton's saturated orange, the square Spectralis infrared field, and different OCT noise textures.

Ask the room: could you tell an OCT from the Triton and the Maestro2 apart? Both are Topcon. Park that thought
for Part 5.

Occasionally a sampled OCT image is almost pure noise; the generator is not perfect. That is also worth saying.
""", minutes=2)

code(r"""
ds = pydicom.dcmread(gallery.local.iloc[0])
for tag in ["SOPClassUID", "Manufacturer", "ManufacturerModelName", "ImageLaterality",
            "Rows", "Columns", "SamplesPerPixel", "PhotometricInterpretation"]:
    value = ds.get(tag)
    print(f"{tag:28s} {getattr(value, 'name', value)}")
""",
notes="""
Real DICOM headers: the device, eye, image size and the standard SOP class are all in the file itself, not just
in the manifest. That means any DICOM tool (viewers, PACS, MONAI, pydicom) works on this data.

Data-quality footnote for the technical crowd: infrared and OCT headers say MONOCHROME1 (which would mean
"invert for display"), but the pixels are stored the normal way round. Do not blindly apply inversion.
""", minutes=0.5, optional=True)

code(r"""
def eye_labels(path):
    obs = pd.read_csv(path, index_col=0, low_memory=False)
    obs["label"] = obs.observation_source_value.str.split(",").str[0].str.replace("mhoccur_", "")
    return obs.pivot_table(index="person_id", columns="label", values="value_as_number")

fundus = fundus.join(eye_labels(paths["fundus/clinical_data/observation.csv"]), on="participant_id")
octs = octs.join(eye_labels(paths["oct/clinical_data/observation.csv"]), on="participant_id")
prevalence = pd.concat({
    "fundus photos": fundus.groupby("manufacturers_model_name")[["amd", "glc", "pdr"]].mean(),
    "OCT": octs.groupby("manufacturers_model_name")[["amd", "glc", "pdr"]].mean(),
}).mul(100).round(1)
prevalence.columns = ["AMD %", "glaucoma %", "diabetic retinopathy %"]
prevalence
""",
notes="""
Every image comes with three disease labels in OMOP format. Prevalence is about 5 percent AMD, 8 percent glaucoma
and 4 percent diabetic retinopathy, and it is nearly identical across devices.

That flatness matters for Part 5: a device classifier cannot cheat by learning that one device sees sicker eyes.
""", minutes=1)

md(r"""
### 3.2 ECG: machine measurements for every participant

Each participant has a 12-lead WFDB record (`.hea` header plus `.dat` samples) from a Philips PageWriter TC30.
In this synthetic release the **waveform samples are flat placeholders**: we checked 301 records and every lead
sits at 0 mV. The synthetic content is in the machine measurements (heart rate, PR, QRS, QT, QTc) and the
device's text interpretation, which appear in each header and in `manifest.tsv`.
""", notes="""
Transition from the eye to the heart. From here on we are in the tabular cohort of 10,518 synthetic
participants, where every file shares the same participant_id.

Say the waveform caveat plainly: anyone who wants to train on raw ECG signals needs the real release. The
measurements are still useful, and Part 4 shows they line up with the wearable data.
""", minutes=0.5)

code(r"""
ecg = pd.read_csv(paths["tabular/cardiac_ecg/manifest.tsv"], sep="\t", index_col=0).set_index("participant_id")

pid = 7
for ext in ("hea", "dat"):
    fetch(f"{ROOT}/tabular/cardiac_ecg/{pid}/{pid}_ecg_synthetic.{ext}")
rec = wfdb.rdrecord(str(CACHE / ROOT / "tabular/cardiac_ecg" / str(pid) / f"{pid}_ecg_synthetic"))
header = dict(line.split(": ", 1) for line in rec.comments if ": " in line)

print(f"participant {pid}: {rec.n_sig} leads ({', '.join(rec.sig_name)}), {rec.fs} Hz, {rec.sig_len / rec.fs:.0f} s")
for key in ["device_model", "interpretation_comment_2", "comment_1_val", "comment_2_val", "comment_3_val", "comment_4_val"]:
    print(f"  {key:26s} {header.get(key)}")
print(f"waveform range across all 12 leads: {np.ptp(rec.p_signal):.2f} mV")

INTERVALS = {"Rate": "heart rate (bpm)", "PR": "PR interval (ms)", "QRSD": "QRS duration (ms)", "QTc": "QTc (ms)"}
fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
for ax, (col, label) in zip(axes, INTERVALS.items()):
    ax.hist(ecg[col], bins=40, color=SLOTS[0], edgecolor=SURFACE, linewidth=0.5)
    ax.set_title(label, loc="left")
    ax.set_yticks([])
    ax.grid(False)
fig.suptitle(f"Machine-measured ECG intervals, {len(ecg):,} synthetic participants", x=0.01, ha="left", fontweight="bold")
plt.tight_layout()
""",
notes="""
WFDB is the PhysioNet format, so the wfdb package reads a record in one line. The header carries the device, the
machine measurements and the device's own text interpretation (for example "Normal sinus rhythm" and an
abnormal flag). The last printed line confirms the flat waveform.

The histograms show the measurements across the whole cohort; the ranges look physiological (heart rate
roughly 40 to 105, QTc centred near 420 ms).

If someone asks: the manifest has a person_id column that is constant; use participant_id.
""", minutes=1)

md(r"""
### 3.3 Wearable and CGM summaries, clinical labs and conditions

The real AI-READI release has 10 days of Dexcom glucose at 5 minute resolution and Garmin streams. The synthetic
release condenses these into **summary features, one row per participant**. The units are not documented inside
the file, so treat them as relative measures.
""", notes="""
Be upfront that the synthetic wearable table is a summary. It is still useful: it is linked to the same synthetic
participants as the labs and ECG, which is what Part 4 needs.
""", minutes=0.5)

code(r"""
wear = pd.read_csv(paths["tabular/synthetic_wearable_activity_glucose.tsv"], sep="\t", index_col=0).set_index("participant_id")

meas = pd.read_csv(paths["tabular/clinical_data/measurement.csv"], index_col=0, low_memory=False)
meas["var"] = meas.measurement_source_value.str.split(",").str[0]
labs = meas.pivot_table(index="person_id", columns="var", values="value_as_number")

cond = pd.read_csv(paths["tabular/clinical_data/condition_occurrence.csv"], index_col=0, low_memory=False)
cond["var"] = cond.condition_source_value.str.split(",").str[0]
cond_names = (cond.groupby("var").condition_source_value.first().str.split(",", n=1).str[1]
              .str.replace(r"\s*\((Examples|elevated)[^)]*\)?$|\s*\([^)]*$", "", regex=True).str.strip())
dx = pd.crosstab(cond.person_id, cond["var"]).clip(upper=1).reindex(labs.index, fill_value=0)

print(f"wearable/CGM: {wear.shape[0]:,} participants x {wear.shape[1]} features")
print(f"labs and vitals: {labs.shape[0]:,} participants x {labs.shape[1]} measures (OMOP measurement)")
print(f"conditions: {dx.shape[1]} self-reported conditions (OMOP condition_occurrence)")
wear.describe().T[["mean", "std", "min", "max"]].round(1)
""",
notes="""
Three tables, one participant key. The OMOP tables are long format (one row per measurement), so we pivot to one
row per person, which is the shape most ML code wants.

Point at blood_glucose_mean and blood_glucose_std: those are the CGM features. heart_rate_mean, stress_mean,
steps and sleep hours come from the Garmin side.
""", minutes=1)

code(r"""
top = dx.mean().mul(100).sort_values().tail(15)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([cond_names.get(v, v) for v in top.index], top.values, color=SLOTS[0], height=0.7)
ax.set_xlabel("% of synthetic participants")
ax.grid(axis="y", visible=False)
ax.set_title("Most common self-reported conditions", loc="left")
plt.tight_layout()
""",
notes="""
The condition list reads like a real T2DM-risk cohort: high blood pressure, high cholesterol, elevated A1C,
arthritis, type 2 diabetes (about 38 percent), obesity and prediabetes lead the list. That is the balanced
recruitment design showing through.
""", minutes=0.5, optional=True)

# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------
md(r"""
## 4. Relationships across data types

Inside the tabular cohort every participant has labs, conditions, an ECG and wearable/CGM features. If the
generator learned the real joint distribution, physiology we already know should show up **across** these files.
That is also a quick way to judge how far to trust a synthetic dataset.
""",
notes="""
Set up the logic: we will use known physiology as a test of the synthetic data. If the ECG heart rate and the
wearable heart rate are unrelated, the generator broke the link and we should not trust subtler findings.
""", minutes=0.5)

code(r"""
cohort = (labs.join(dx)
              .join(wear.add_prefix("wear_"))
              .join(ecg[["Rate", "PR", "QRSD", "QT", "QTc"]].add_prefix("ecg_")))

VARS = {  # column: (label, source)
    "import_hba1c": ("HbA1c", "lab"), "import_glucose": ("serum glucose", "lab"),
    "import_triglycerides": ("triglycerides", "lab"), "import_hdl_cholesterol": ("HDL cholesterol", "lab"),
    "bmi_vsorres": ("BMI", "vitals"), "waist_vsorres": ("waist", "vitals"), "bp1_sysbp_vsorres": ("systolic BP", "vitals"),
    "mhterm_dm2": ("type 2 diabetes", "condition"), "mhoccur_hbp": ("high blood pressure", "condition"),
    "wear_blood_glucose_mean": ("CGM mean glucose", "CGM"), "wear_blood_glucose_std": ("CGM glucose SD", "CGM"),
    "wear_heart_rate_mean": ("mean heart rate", "wearable"), "wear_resting_heart_rate": ("resting heart rate", "wearable"),
    "wear_stress_mean": ("stress score", "wearable"), "wear_total_steps": ("steps", "wearable"),
    "ecg_Rate": ("heart rate", "ECG"), "ecg_QT": ("QT", "ECG"), "ecg_QTc": ("QTc", "ECG"),
    "moca_total_score": ("MoCA total", "cognition"),
}
corr = cohort[list(VARS)].corr(method="spearman")
names = [f"{label} ({src})" for label, src in VARS.values()]

diverging = sns.blend_palette(["#104281", "#3987e5", "#f0efec", "#e66767", "#a61e1e"], as_cmap=True)
annot = corr.map(lambda r: f"{r:.2f}" if abs(r) >= 0.2 and abs(r) < 0.999 else "")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap=diverging, vmin=-1, vmax=1, center=0, annot=annot, fmt="", annot_kws={"size": 7},
            xticklabels=names, yticklabels=names, linewidths=1, linecolor=SURFACE,
            cbar_kws={"label": "Spearman correlation", "shrink": 0.6}, ax=ax)
ax.grid(False)
ax.set_title(f"Correlations across labs, conditions, CGM, wearable and ECG (n = {len(cohort):,})", loc="left")
plt.tight_layout()
""",
notes="""
Let the heatmap sit for a moment, then walk three blocks (numbers are shown only where the correlation is at
least 0.2):

1. Glycemia block: HbA1c, serum glucose, CGM mean glucose and type 2 diabetes all move together. The CGM file and
   the lab file are different tables, so this link survived synthesis.
2. Heart block: ECG heart rate against wearable mean and resting heart rate is about 0.7. Two different devices,
   two different files, consistent physiology.
3. ECG internal: QT shortens as heart rate rises (negative), which is textbook.

Then point at the blanks: steps against BMI is near zero, and systolic blood pressure barely relates to the
high blood pressure diagnosis. Section 4.2 makes that explicit.
""", minutes=2)

code(r"""
checks = [  # (x, y, what physiology predicts)
    ("ecg_Rate", "wear_heart_rate_mean", "strong positive"),
    ("ecg_QT", "ecg_Rate", "strong negative"),
    ("import_hba1c", "wear_blood_glucose_mean", "positive"),
    ("import_hba1c", "mhterm_dm2", "positive"),
    ("bmi_vsorres", "waist_vsorres", "strong positive"),
    ("bp1_sysbp_vsorres", "mhoccur_hbp", "positive"),
    ("wear_total_steps", "bmi_vsorres", "negative"),
    ("import_creatinine", "mhoccur_rnl", "positive"),
]
pd.DataFrame([{
    "pair": f"{a}  vs  {b}", "expected": e,
    "spearman r": round(cohort[a].corr(cohort[b], method="spearman"), 2)} for a, b, e in checks
]).set_index("pair")
""",
notes="""
This is the fidelity scorecard in miniature. The first five rows behave as expected. The last three (blood
pressure vs hypertension diagnosis, steps vs BMI, creatinine vs kidney problems) are much weaker than you would
see in real people.

Message: synthetic data preserved the strong, common relationships and flattened some weaker or rarer ones.
Use it to build and debug pipelines, then confirm on the real release. Measuring this gap systematically is
itself a research question (Part 6).
""", minutes=1)

code(r"""
t2d = cohort.assign(group=cohort.mhterm_dm2.map({0: "no T2DM", 1: "T2DM"}))
panels = [("import_hba1c", "HbA1c (%)"), ("wear_blood_glucose_mean", "CGM mean glucose"), ("bmi_vsorres", "BMI")]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (col, label) in zip(axes, panels):
    sns.boxplot(data=t2d, x="group", y=col, hue="group", order=["no T2DM", "T2DM"],
                palette={"no T2DM": SLOTS[0], "T2DM": SLOTS[1]}, showfliers=False, width=0.5, ax=ax, legend=False)
    means = t2d.groupby("group")[col].mean()
    ax.set_title(f"{label}: {means['no T2DM']:.1f} vs {means['T2DM']:.1f}", loc="left")
    ax.set_xlabel(""); ax.set_ylabel(label)
plt.tight_layout()
""",
notes="""
Same story from the diagnosis side. People who report type 2 diabetes have higher HbA1c (about 6.4 vs 5.5
percent), higher CGM mean glucose and higher BMI. Three different tables, one consistent picture.

Good moment to ask the room what they would do with a real version of this: for example, which CGM patterns
separate insulin users from people controlled by lifestyle alone.
""", minutes=1)

code(r"""
fig, ax = plt.subplots(figsize=(5.5, 5))
hb = ax.hexbin(cohort.ecg_Rate, cohort.wear_heart_rate_mean, gridsize=45, mincnt=1,
               cmap=sns.blend_palette(["#cde2fb", "#3987e5", "#0d366b"], as_cmap=True))
r = cohort.ecg_Rate.corr(cohort.wear_heart_rate_mean)
ax.set_xlabel("ECG heart rate, clinic (bpm)")
ax.set_ylabel("wearable mean heart rate (bpm)")
ax.set_title(f"Two devices, one heart: r = {r:.2f}", loc="left")
fig.colorbar(hb, ax=ax, label="participants")
plt.tight_layout()
""",
notes="""
One scatter to make the cross-device link concrete: a 10 second clinic ECG against days of wrist-worn heart rate.
Optional if short on time.
""", minutes=0.5, optional=True)

# ---------------------------------------------------------------------------
# 5. Device challenge
# ---------------------------------------------------------------------------
md(r"""
## 5. Challenge: which device took this scan?

Every imaging device leaves a fingerprint: optics, sensor noise, compression, post-processing. In a famous chest
X-ray study, a network could tell *which hospital* an image came from almost perfectly
([Zech et al., PLOS Medicine 2018](https://doi.org/10.1371/journal.pmed.1002683)). Fingerprints like this are how
models learn shortcuts and why a model trained on one OCT device often underperforms on another.

Last year's AI-READI demo showed that a classifier can name the device from grayscale random pieces of retinal
images that people cannot tell apart. We repeat it here on **synthetic** OCT, which asks an extra question:
**did the diffusion model learn each device's fingerprint too?**

We make the task harder step by step:

| Level | What the model sees |
|---|---|
| `whole` | the full B-scan, shrunk to 64 x 64 pixels |
| `crop64` | a random 64 x 64 piece at full resolution |
| `crop64n` | the same piece with brightness and contrast removed |
| `crop32n` | a random 32 x 32 piece, brightness and contrast removed |

Four devices, so guessing gets 25%.
""",
notes="""
This is the main event. Credit Aaron Lee's demo from last year: people were shown grayscale random crops of
retinal images and asked to name the device; they could not, the classifier could.

Why it matters: if a model can tell the device from a tiny patch, then any disease model trained
on images can silently use the device as a proxy (for example if one clinic with sicker patients uses one
machine). That is the domain shift problem that holds back deployment.

Twist for the synthetic version: the generator was told the device for every image. If a classifier recovers the
device, the generator reproduced the fingerprint, which is a nontrivial test of how realistic the images are.
""", minutes=2)

code(r"""
MODALITY = "OCT"        # try "Color Photography" or "Infrared Reflectance" afterwards
N_PER_DEVICE = 1000     # images per device; 1,000 is about 65 MB for OCT

pool = images[images.imaging == MODALITY]
subset = pool.groupby("device").sample(N_PER_DEVICE, random_state=SEED).reset_index(drop=True)
t0 = time.time()
subset["local"] = fetch_many(subset.key)

def load_gray(path):
    px = read_pixels(path)
    if px.ndim == 3:  # RGB photo: convert to grayscale, as in the original demo
        px = px[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return px.astype(np.uint8)

with ThreadPoolExecutor(16) as pool_:
    X = np.stack(list(pool_.map(load_gray, subset.local)))
DEVICES = sorted(subset.device.unique())
y = subset.device.map({d: i for i, d in enumerate(DEVICES)}).to_numpy()
print(f"{len(X):,} {MODALITY} images {X.shape[1:]} from {DEVICES} in {time.time() - t0:.0f} s")
""",
notes="""
Pulls a balanced sample, 1,000 B-scans per device, directly from S3 with 32 threads. In-region this takes
well under a minute. Color photos get converted to grayscale, which was the original trick for removing the
obvious color cues.

Every image belongs to a different synthetic participant, so a random train/test split has no patient leakage.
""", minutes=1)

code(r"""
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count())
Xt = torch.from_numpy(X)                                 # uint8; converted to float per batch to save memory
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=SEED)
LEVELS = ["whole", "crop64", "crop64n", "crop32n"]

def make_view(idx, level, seed):
    # One batch of model inputs for a difficulty level.
    g = torch.Generator().manual_seed(seed)
    batch = Xt[idx]
    if level == "whole":                                  # full scan, 256 -> 64 pixels
        return torch.cat([F.avg_pool2d(c[:, None].float() / 255, 4) for c in batch.split(500)])
    size = 32 if level == "crop32n" else 64
    top = torch.randint(0, 257 - size, (len(idx),), generator=g)
    left = torch.randint(0, 257 - size, (len(idx),), generator=g)
    crops = torch.stack([im[t:t + size, l:l + size] for im, t, l in zip(batch, top, left)])[:, None].float() / 255
    if level.endswith("n"):                              # remove brightness and contrast cues
        crops = (crops - crops.mean((2, 3), keepdim=True)) / (crops.std((2, 3), keepdim=True) + 1e-6)
    return crops

fig, axes = plt.subplots(len(DEVICES), len(LEVELS), figsize=(9, 9))
example = [int(train_idx[y[train_idx] == k][0]) for k in range(len(DEVICES))]
for i, idx in enumerate(example):
    for j, level in enumerate(LEVELS):
        ax = axes[i, j]
        ax.imshow(make_view([idx], level, seed=j)[0, 0], cmap="gray")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if i == 0: ax.set_title(level)
        if j == 0: ax.set_ylabel(DEVICES[i], rotation=0, ha="right", va="center")
plt.suptitle("What each level shows: one example per device", x=0.01, ha="left", fontweight="bold")
plt.tight_layout()
""",
notes="""
Walk left to right. With the whole scan you can see layer shape and background. A 64 pixel crop keeps some
texture. Once brightness and contrast are removed, most people are guessing. At 32 pixels there is almost
nothing a human can use.
""", minutes=1)

md(r"""
### 5.1 Your turn

Below are 12 contrast-normalized 64 x 64 crops (level `crop64n`) from scans held out of training. Study the
labeled row first, then write down a device letter for each numbered crop.
""", notes="""
Give the room about a minute. If people are following along, they type their answer string into the next cell;
otherwise ask for a show of hands on two or three crops.
""", minutes=0.5)

code(r"""
LETTERS = {d: chr(65 + i) for i, d in enumerate(DEVICES)}
rng = np.random.default_rng(2026)
quiz_idx = rng.permutation(np.concatenate(
    [rng.choice(test_idx[y[test_idx] == k], 3, replace=False) for k in range(len(DEVICES))]))
quiz_views = make_view(quiz_idx, "crop64n", seed=2026)

fig = plt.figure(figsize=(12, 5.6))
grid = fig.add_gridspec(3, 8, hspace=0.35)
for k, d in enumerate(DEVICES):                          # labeled study examples: two per device
    for s in range(2):
        ax = fig.add_subplot(grid[0, 2 * k + s])
        idx = int(train_idx[y[train_idx] == k][10 + s])
        ax.imshow(make_view([idx], "crop64n", seed=s)[0, 0], cmap="gray")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if s == 0: ax.set_title(f"{LETTERS[d]} = {d}", loc="left")
for q in range(12):                                     # the quiz
    ax = fig.add_subplot(grid[1 + q // 6, (q % 6) + 1])
    ax.imshow(quiz_views[q, 0], cmap="gray")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"#{q + 1}")
fig.suptitle("Top row: labeled examples. Rows below: which device? Answer with letters.",
             x=0.01, ha="left", fontweight="bold")
plt.show()
""",
notes="""
The top row is the "training data" for humans: two labeled crops per device. The bottom two rows are the quiz.
The answer key is revealed in cell 5.3 after the model has trained, so do not scroll ahead.
""", minutes=1.5)

code(r"""
MY_GUESSES = ""   # e.g. "ABDCCABDAACB", one letter per numbered crop
print(f"{len(MY_GUESSES.strip())} of 12 answers entered")
""",
notes="""
Hands-on attendees fill this in. If nobody does, it is scored as an empty answer and the reveal still works.
""", minutes=0.5)

md(r"""
### 5.2 Now the model

A four-layer convolutional network, trained from scratch on each level with fresh random crops every epoch. On a
GPU the four levels take under a minute; on a CPU-only VM, a few minutes.
""", notes="""
While the next cell runs, talk through what the network is: about 100 thousand parameters, far smaller than any
modern vision model, no pretraining. A model this small doing this well tells you the fingerprint is easy to
learn.
""", minutes=0.5)

code(r"""
EPOCHS = 12 if DEV == "cuda" else 6

def small_cnn(n_classes):
    layers, ch = [], [1, 32, 64, 128]
    for a, b in zip(ch[:-1], ch[1:]):
        layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes))

def train_level(level, epochs=EPOCHS, batch=128):
    torch.manual_seed(SEED)
    model = small_cnn(len(DEVICES)).to(DEV)
    steps = epochs * ((len(train_idx) + batch - 1) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps)
    targets = torch.from_numpy(y[train_idx])
    for epoch in range(epochs):
        model.train()
        views = make_view(train_idx, level, seed=epoch)  # new random crops every epoch
        for b in torch.randperm(len(train_idx)).split(batch):
            if len(b) < 2:                               # BatchNorm needs at least two samples
                continue
            loss = F.cross_entropy(model(views[b].to(DEV)), targets[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

@torch.no_grad()
def predict(model, views):
    model.eval()
    return torch.cat([model(v.to(DEV)).argmax(1).cpu() for v in views.split(512)]).numpy()

models, accuracy = {}, {}
for level in LEVELS:
    t0 = time.time()
    models[level] = train_level(level)
    pred = predict(models[level], make_view(test_idx, level, seed=999))
    accuracy[level] = (pred == y[test_idx]).mean()
    print(f"{level:8s} test accuracy {accuracy[level]:.1%}   ({time.time() - t0:.0f} s on {DEV})")
""",
notes="""
Expected results from our dry run on OCT (GPU, 12 epochs): whole about 99 percent, crop64 about 92, crop64n
about 90, crop32n about 80. On CPU with 6 epochs the harder levels land a few points lower. Chance is 25.

If time is short, set LEVELS to ["whole", "crop64n"] before running; the story still works.
""", minutes=3)

code(r"""
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(LEVELS, [accuracy[l] * 100 for l in LEVELS], color=SLOTS[0], width=0.6)
ax.bar_label(bars, labels=[f"{accuracy[l]:.1%}" for l in LEVELS], padding=3, color=INK)
ax.axhline(100 / len(DEVICES), color=INK2, linestyle="--", linewidth=1)
ax.text(len(LEVELS) - 0.5, 100 / len(DEVICES) + 2, "chance", ha="right", color=INK2)
ax.set_ylim(0, 105); ax.set_ylabel("test accuracy (%)"); ax.grid(axis="x", visible=False)
ax.set_title(f"Naming the {MODALITY} device as the task gets harder", loc="left")
plt.tight_layout()
""",
notes="""
The headline chart. Even a 32 pixel patch with brightness and contrast removed is identified far above chance.
The model is reading texture: speckle and noise structure, not anatomy.
""", minutes=1)

code(r"""
from sklearn.metrics import confusion_matrix
level = "crop64n"
pred = predict(models[level], make_view(test_idx, level, seed=999))
cm = confusion_matrix(y[test_idx], pred, normalize="true") * 100
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt=".0f", cmap=sns.blend_palette(["#f0efec", "#3987e5", "#0d366b"], as_cmap=True),
            xticklabels=DEVICES, yticklabels=DEVICES, cbar=False, linewidths=1, linecolor=SURFACE, ax=ax)
ax.grid(False)
ax.set_xlabel("predicted device"); ax.set_ylabel("true device")
ax.set_title(f"Where the model gets confused ({level}, % of each row)", loc="left")
plt.tight_layout()
""",
notes="""
Look for the biggest off-diagonal cells. In our runs the Maestro2 and the Triton are confused most often, and
they are both Topcon devices. Same manufacturer, similar processing pipeline, similar fingerprint. Spectralis is
the easiest to spot.
""", minutes=1)

md(r"""
### 5.3 The reveal
""", notes="""
Ask everyone to lock in their answers before running the next cell.
""", minutes=0.25)

code(r"""
quiz_pred = predict(models["crop64n"], quiz_views)
answer = pd.DataFrame({
    "crop": [f"#{q + 1}" for q in range(12)],
    "true device": [f"{LETTERS[DEVICES[k]]} {DEVICES[k]}" for k in y[quiz_idx]],
    "model": [f"{LETTERS[DEVICES[k]]} {DEVICES[k]}" for k in quiz_pred],
}).set_index("crop")
answer["you"] = list(MY_GUESSES.upper().ljust(12, "-"))[:12]
model_score = (quiz_pred == y[quiz_idx]).sum()
your_score = sum(g == t[0] for g, t in zip(answer["you"], answer["true device"]))
print(f"model: {model_score}/12   you: {your_score}/12   chance: 3/12")
answer
""",
notes="""
Read out the model's score, then ask who beat it. Almost nobody does. That is the point: the fingerprint is real
(here, reproduced by the generator), invisible to us, and trivially learnable by a machine.
""", minutes=1.5)

code(r"""
def radial_spectrum(views):
    # Average power at each spatial frequency, over a batch of normalized crops.
    n = views.shape[-1]
    win = torch.hann_window(n)[:, None] * torch.hann_window(n)[None, :]
    power = torch.fft.fftshift(torch.fft.fft2(views[:, 0] * win), dim=(-2, -1)).abs().pow(2).mean(0)
    yy, xx = torch.meshgrid(torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij")
    radius = (yy ** 2 + xx ** 2).sqrt().round().long().ravel()
    sums = torch.zeros(radius.max() + 1).index_add_(0, radius, power.ravel())
    return (sums / torch.bincount(radius))[1: n // 2].numpy()

fig, ax = plt.subplots(figsize=(8, 4.5))
freqs = np.arange(1, 32)
spectra = {d: np.log10(radial_spectrum(make_view(test_idx[y[test_idx] == k], "crop64n", seed=7)))
           for k, d in enumerate(DEVICES)}
span = max(s.max() for s in spectra.values()) - min(s.min() for s in spectra.values())
label_y, gap = {}, 0.04 * span                           # direct labels, nudged apart so they never overlap
for d in sorted(spectra, key=lambda d: spectra[d][-1]):
    label_y[d] = max(spectra[d][-1], max(label_y.values(), default=-np.inf) + gap)
for k, d in enumerate(DEVICES):
    ax.plot(freqs, spectra[d], color=SLOTS[k])
    ax.text(freqs[-1] + 0.6, label_y[d], d, color=INK2, va="center")
ax.set_xlim(1, 37)
ax.set_xlabel("spatial frequency (cycles per 64 pixel crop)  ->  finer texture")
ax.set_ylabel("log10 power")
ax.set_title("Why the model can tell: each device has its own texture spectrum", loc="left")
plt.tight_layout()
""",
notes="""
The explanation slide. After removing brightness and contrast, the remaining difference is how much fine-grained
texture each device has. Spectralis images are smoother (power drops fastest at high frequency), Maestro2 is the
grainiest. A human eye does not measure this; a convolution does it in its first layer.

Optional: rerun Part 5 with MODALITY = "Color Photography". Grayscale color photos are about 98 percent
identifiable from the whole image and still well above chance from crops.
""", minutes=1.5, optional=True)

# ---------------------------------------------------------------------------
# 6. Grant ideas
# ---------------------------------------------------------------------------
md(r"""
## 6. Problems worth a grant

Each idea below can be prototyped on the synthetic data today and then run on the real AI-READI release.

**1. Device-invariant retinal biomarkers.** Part 5 shows how easily the device leaks into a model. Build
representations that predict disease but *cannot* predict the device, and use the device classifier from this
notebook as the leakage test. AI-READI is one of the few datasets where the same eyes are imaged on several
machines, which makes this testable. (Related: the MICCAI 2026 DA-OCT cross-vendor OCT challenge on AI-READI.)

**2. The retina as a window on metabolic health.** On the real data, images and labs belong to the same people:
predict HbA1c, kidney function or diabetes severity group from retinal photos and OCT, and ask whether the retina
shows who is moving toward health. Build and debug the pipeline on synthetic data now.

**3. Glucose dynamics in daily life.** Link CGM variability to sleep, stress, activity and air quality at home.
The real release has 10 days of 5 minute CGM plus Garmin and environmental streams on the same people; the
synthetic summaries already show glycemic signals lining up with HbA1c and diagnosis.

**4. Pathways toward health (salutogenesis).** The four severity groups from no diabetes to insulin form a
spectrum. Model where individuals sit on it across eyes, heart, glucose and cognition, and which factors mark
the people doing better than their group.

**5. How faithful is synthetic multimodal data?** Part 4 found that strong relationships survived synthesis
and several weaker ones did not. A systematic fidelity and privacy scorecard for multimodal synthetic health
data, validated against the real AI-READI release, would help every Bridge2AI dataset.
""",
notes="""
Pick two to discuss depending on the audience. For an imaging audience lead with 1 and 2; for a clinical
audience lead with 3 and 4; for informatics and privacy people lead with 5.

Why these are strong grant angles: each uses something rare about AI-READI (multi-device imaging of the same
people, per-person multimodality, deliberate balance across severity and race and ethnicity), and each has a
synthetic-data prototype path, which lowers risk in a proposal.

Invite the room: what would you add?
""", minutes=4)

md(r"""
## 7. Next steps and resources

* **Real AI-READI data:** documentation at [docs.aireadi.org](https://docs.aireadi.org/), access through
  [FAIRhub](https://fairhub.io/) under a data license agreement.
* **This synthetic dataset:** AWS Marketplace listing *ADVANCE Center Synthetic AI-READI Dataset for T2DM Research*;
  archive and license on [Zenodo (doi:10.5281/zenodo.22944362)](https://doi.org/10.5281/zenodo.22944362); generation
  code at [github.com/nicholas-j-jackson/ai_readi_synth](https://github.com/nicholas-j-jackson/ai_readi_synth).
* **Citing AI-READI:** AI-READI Consortium, *Nature Metabolism* 2024,
  [doi:10.1038/s42255-024-01165-x](https://doi.org/10.1038/s42255-024-01165-x); study protocol, *BMJ Open* 2025,
  [doi:10.1136/bmjopen-2024-097449](https://doi.org/10.1136/bmjopen-2024-097449).
* **Device fingerprints and shortcut learning:** Zech et al., *PLOS Medicine* 2018,
  [doi:10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683).

Try it yourself: change `MODALITY` in Part 5 to `"Color Photography"` or `"Infrared Reflectance"`, or swap the
variables in Part 4 for your own hypothesis.
""",
notes="""
Close by repeating the two caveats (synthetic images are not linked to the tabular participants; confirm
findings on real data) and the one invitation (pick a problem from Part 6 and start prototyping this week).
""", minutes=1)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def heading(src):
    for line in src.splitlines():
        if line.startswith("#") and not line.startswith("#!") and line.lstrip("#").startswith(" "):
            return line.lstrip("#").strip()
    return None


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    notes_md = ["# Presenter notes: AI-READI synthetic dataset demo", "",
                "Generated from `build_notebook.py`; cell numbers match the notebook top to bottom. "
                "Minutes are rough speaking time. Cells marked optional can be skipped if running long.", ""]
    section, total = None, 0.0
    for n, c in enumerate(CELLS, start=1):
        if c["kind"] == "md":
            cell = nbf.v4.new_markdown_cell(c["src"])
            section = heading(c["src"]) or section
        else:
            cell = nbf.v4.new_code_cell(c["src"])
        if c["notes"]:
            cell.metadata["presenter_notes"] = c["notes"]
        if c["optional"]:
            cell.metadata["tags"] = ["optional"]
        nb.cells.append(cell)
        if not c["notes"]:
            continue
        mins = c["minutes"] or 0
        total += mins
        label = heading(c["src"]) if c["kind"] == "md" else f"code, in {section}"
        extra = [f"~{mins:g} min"] if mins else []
        if c["optional"]:
            extra.append("optional")
        notes_md.append(f"## Cell {n}: {label}" + (f" ({', '.join(extra)})" if extra else ""))
        notes_md.append("")
        notes_md.append(c["notes"])
        notes_md.append("")
    notes_md.insert(3, f"Total if every cell is presented: about {total:.0f} minutes.\n")
    nbf.write(nb, NB_PATH)
    NOTES_PATH.write_text("\n".join(notes_md))
    print(f"wrote {NB_PATH.name} ({len(nb.cells)} cells) and {NOTES_PATH.name} (~{total:.0f} min)")


if __name__ == "__main__":
    build()
