#!/usr/bin/env bash
# Copy the notebook and sample data to the workshop storage folder, then build the participant and hub images.
set -euo pipefail
cd "$(dirname "$0")"

STORAGE="${WORKSHOP_STORAGE:-/mnt/fast/aireadi-workshop}"
case "$STORAGE" in /?*) ;; *) echo "WORKSHOP_STORAGE must be an absolute path" >&2; exit 1 ;; esac
if [ ! -d ../data/scans ]; then
  echo "data folder not found next to the notebook; unzip aireadi_workshop_data.zip in the repository folder first" >&2
  exit 1
fi

mkdir -p "$STORAGE/data" "$STORAGE/template" "$STORAGE/users"
chmod -R u+w "$STORAGE/data"                     # allow refreshing the copy on a second run
cp -a ../data/. "$STORAGE/data/"
cp ../aireadi_synthetic_demo.ipynb "$STORAGE/template/"
chmod -R a-w "$STORAGE/data"
echo "workshop files in $STORAGE: $(find "$STORAGE/data/scans" -name '*.png' | wc -l) scans, notebook template, users/"

docker build -f Dockerfile.user -t aireadi-workshop-user:latest .
docker compose build
echo "built aireadi-workshop-user:latest and aireadi-workshop-hub:latest"
