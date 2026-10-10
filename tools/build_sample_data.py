"""Build the small sample dataset in ../data from the synthetic AI-READI bucket.

The workshop notebook only reads ../data. This script is how that folder was made, so anyone with
their own access key can rebuild the identical sample (sampling uses fixed seeds).

    # with temporary AWS keys exported, or credentials.csv in the working directory
    python tools/build_sample_data.py

    # or from a local copy of the bucket (same key layout, starting with "ai-readi/")
    python tools/build_sample_data.py --from-dir /path/to/mirror

Output (about 85 MB):
    data/scans/scan_0001.png ...   2,000 retina cross-section scans, 500 per machine, lossless PNG
    data/scans.csv                 which machine took each scan, plus the original file name
    data/gallery/*.png             one example per kind of image and machine (13 files)
    data/gallery.csv
    data/participants.csv          16 measurements for 10,518 synthetic participants
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

ROOT = "ai-readi"
SCANS_PER_MACHINE = 500
SEED = 42
OUT = Path(__file__).resolve().parent.parent / "data"

KIND = {"Color Photography": "color photo", "Infrared Reflectance": "infrared photo",
        "Autofluorescence": "autofluorescence photo", "OCT": "cross-section scan"}

# source column -> plain name used in participants.csv
LABS = {
    "import_hba1c": "hba1c_percent",
    "import_glucose": "lab_glucose_mg_dl",
    "import_creatinine": "creatinine_mg_dl",
    "bmi_vsorres": "body_mass_index",
    "waist_vsorres": "waist_cm",
    "bp1_sysbp_vsorres": "systolic_blood_pressure_mmhg",
}
CONDITIONS = {"mhterm_dm2": "type2_diabetes", "mhoccur_hbp": "high_blood_pressure", "mhoccur_rnl": "kidney_problems"}
WEARABLE = {
    "blood_glucose_mean": "glucose_monitor_mean_mg_dl",
    "blood_glucose_std": "glucose_monitor_variability_mg_dl",
    "heart_rate_mean": "watch_heart_rate_bpm",
    "resting_heart_rate": "watch_resting_heart_rate_bpm",
    "total_steps": "watch_steps",
    "stress_mean": "watch_stress_score",
}
ECG = {"Rate": "clinic_heart_rate_bpm"}


class Source:
    """Reads keys from S3, or from a local directory laid out like the bucket."""

    def __init__(self, from_dir=None, bucket="ai-readi-sd", region="us-east-2", cache="~/aireadi_cache"):
        self.from_dir = Path(from_dir).expanduser() if from_dir else None
        self.cache = Path(cache).expanduser()
        if not self.from_dir:
            import boto3
            import botocore
            if not os.environ.get("AWS_ACCESS_KEY_ID") and Path("credentials.csv").exists():
                row = next(csv.DictReader(open("credentials.csv")))
                os.environ.update(AWS_ACCESS_KEY_ID=row["access_key"], AWS_SECRET_ACCESS_KEY=row["secret_key"],
                                  AWS_SESSION_TOKEN=row["session_token"])
            self.bucket = bucket
            self.s3 = boto3.client("s3", region_name=region,
                                   config=botocore.config.Config(max_pool_connections=64))

    def path(self, key):
        if self.from_dir:
            return self.from_dir / key
        dst = self.cache / key
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, key, str(dst))
        return dst

    def paths(self, keys):
        with ThreadPoolExecutor(32) as pool:
            return list(pool.map(self.path, keys))


def image_manifest(src):
    fundus = pd.read_csv(src.path(f"{ROOT}/fundus/retinal_photography/manifest.tsv"), sep="\t", index_col=0)
    octs = pd.read_csv(src.path(f"{ROOT}/oct/retinal_oct/manifest.tsv"), sep="\t", index_col=0)
    fundus["key"] = f"{ROOT}/fundus/" + fundus.filepath
    octs["key"] = f"{ROOT}/oct/" + octs.filepath
    octs["imaging"] = "OCT"
    images = pd.concat([fundus, octs], ignore_index=True)
    images["device"] = images.manufacturers_model_name
    return images


def save_png(dicom_path, png_path):
    Image.fromarray(pydicom.dcmread(dicom_path).pixel_array).save(png_path, optimize=True)


def build_scans(src, images):
    pool = images[images.imaging == "OCT"]
    sample = pool.groupby("device").sample(SCANS_PER_MACHINE, random_state=SEED)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle so file order hides the machine
    sample["file"] = [f"scan_{i + 1:04d}.png" for i in range(len(sample))]
    (OUT / "scans").mkdir(parents=True, exist_ok=True)
    local = src.paths(sample.key)
    with ThreadPoolExecutor(16) as pool_:
        list(pool_.map(save_png, local, [OUT / "scans" / f for f in sample.file]))
    table = pd.DataFrame({
        "file": sample.file, "machine": sample.device, "maker": sample.manufacturer,
        "area": sample.anatomic_region.str.lower(), "eye": sample.laterality.map({"L": "left", "R": "right"}),
        "source_key": sample.key,
    })
    table.to_csv(OUT / "scans.csv", index=False)
    print(f"scans: {len(table)} files, {table.machine.value_counts().to_dict()}")


def build_gallery(src, images):
    pick = images.groupby(["imaging", "device"]).sample(1, random_state=SEED)
    (OUT / "gallery").mkdir(parents=True, exist_ok=True)
    rows = []
    for (_, r), path in zip(pick.iterrows(), src.paths(pick.key)):
        name = f"{KIND[r.imaging].replace(' ', '_')}_{r.device.lower()}.png"
        save_png(path, OUT / "gallery" / name)
        rows.append({"file": name, "kind": KIND[r.imaging], "machine": r.device, "maker": r.manufacturer,
                     "source_key": r.key})
    pd.DataFrame(rows).to_csv(OUT / "gallery.csv", index=False)
    print(f"gallery: {len(rows)} files")


def build_participants(src):
    meas = pd.read_csv(src.path(f"{ROOT}/tabular/clinical_data/measurement.csv"), index_col=0, low_memory=False)
    meas["var"] = meas.measurement_source_value.str.split(",").str[0]
    labs = meas.pivot_table(index="person_id", columns="var", values="value_as_number")[list(LABS)].rename(columns=LABS)

    cond = pd.read_csv(src.path(f"{ROOT}/tabular/clinical_data/condition_occurrence.csv"), index_col=0, low_memory=False)
    cond["var"] = cond.condition_source_value.str.split(",").str[0]
    dx = (pd.crosstab(cond.person_id, cond["var"]).clip(upper=1)
            .reindex(index=labs.index, columns=list(CONDITIONS), fill_value=0).rename(columns=CONDITIONS))

    wear = pd.read_csv(src.path(f"{ROOT}/tabular/synthetic_wearable_activity_glucose.tsv"), sep="\t", index_col=0)
    wear = wear.set_index("participant_id")[list(WEARABLE)].rename(columns=WEARABLE)
    ecg = pd.read_csv(src.path(f"{ROOT}/tabular/cardiac_ecg/manifest.tsv"), sep="\t", index_col=0)
    ecg = ecg.set_index("participant_id")[list(ECG)].rename(columns=ECG)

    table = dx.join(labs).join(wear).join(ecg).round(2)
    table.index.name = "participant_id"
    table.to_csv(OUT / "participants.csv")
    print(f"participants: {table.shape[0]} rows x {table.shape[1]} columns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", help="local directory laid out like the bucket (keys start with ai-readi/)")
    args = ap.parse_args()
    src = Source(args.from_dir)
    images = image_manifest(src)
    build_scans(src, images)
    build_gallery(src, images)
    build_participants(src)
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"done: {OUT} ({total:.0f} MB)")
