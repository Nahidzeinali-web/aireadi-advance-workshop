# AI-READI ADVANCE workshop

Notebook demo for the October 21, 2026 AI-READI session. It walks through the ADVANCE Center Synthetic AI-READI
Dataset for T2DM Research on S3: an introduction to AI-READI, a tour of every data type, relationships across
data types, a "which device took this scan?" challenge, and research ideas.

| File | What it is |
|---|---|
| `aireadi_synthetic_demo.ipynb` | the notebook (presenter notes are also stored in each cell's metadata) |
| `PRESENTER_NOTES.md` | talking points per cell, with rough timings; cell numbers match the notebook |
| `build_notebook.py` | source of truth for both files above. Edit this, then run `python build_notebook.py`; that writes a notebook without outputs, so run it top to bottom again before committing |
| `requirements.txt` | Python packages |

## Running it on a cloud VM

1. Start a VM in **us-east-2** (the `ai-readi-sd` bucket) or **us-east-1** (the identical `ai-readi-bucket`
   mirror; set `AIREADI_BUCKET=ai-readi-bucket` and `AIREADI_REGION=us-east-1`). 4 or more vCPUs and 16 GB RAM
   is enough. A GPU is optional; without one, Part 5 takes a few minutes instead of under one.
2. Install: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
3. Credentials: download a fresh credentials ZIP **right before the session**. The instructions say the keys
   last 12 hours, but in testing on 2026-09-29 they expired after about one hour. Either export
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`, or put `credentials.csv` next to the
   notebook (or point `AIREADI_CREDS_CSV` at it). Never commit or share that file.
4. `jupyter lab`, open the notebook, run top to bottom.

Downloads are cached in `~/aireadi_cache` (override with `AIREADI_CACHE`), about 550 MB for a full run, so a rerun
after a credential refresh does not download anything twice.
