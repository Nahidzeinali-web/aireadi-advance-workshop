# AI-READI ADVANCE workshop

A hands-on notebook for AI-READI workshop. It uses a small sample of the ADVANCE Center synthetic AI-READI dataset and runs offline on a laptop
processor; no cloud account is needed.

| File | What it is |
|---|---|
| `aireadi_synthetic_demo.ipynb` | the notebook, saved with every cell's output |
| `PRESENTER_NOTES.md` | talking points and rough timings for each cell; cell numbers match the notebook |
| `build_notebook.py` | source for both files above |
| `data/README.md`, `data/LICENSE-AI-READI-synthetic-data.pdf` | what the sample contains, and the license that governs it |
| `tools/build_sample_data.py` | builds the `data` folder from the dataset with your own access key |

## Computer requirements

| | Needed | Comfortable |
|---|---|---|
| Processor | 2 cores | 4 or more cores |
| Memory | 4 GB | 8 GB |
| Free disk space | 2 GB (software about 1.5 GB, data 75 MB) | |
| Software | Python 3.10 or newer | |
| Time to run every cell | about 5 minutes on 4 cores | |

No graphics card is needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # processor-only build, much smaller
pip install -r requirements.txt
jupyter lab                                  # open aireadi_synthetic_demo.ipynb
```

## Getting the data

The sample data is **not stored in this repository**. The AI-READI Synthetic Data License Agreement (Licensor:
Washington University in St. Louis) covers only the people a license was granted to, and does not allow posting the
data as a download. See `data/README.md` for the details.

* If you are covered by our group license, get `aireadi_workshop_data.zip` from the workshop organizers and unzip
  it next to the notebook. It creates the `data` folder, including the license text and a description of the
  files.
* Or build it yourself: request access on the Synthetix portal of the ADVANCE Center
  (https://hiplab.vumc.org/synthetix), download your credentials ZIP, put its `credentials.csv` in this folder, and
  run:

  ```bash
  pip install -r tools/requirements.txt
  python tools/build_sample_data.py          # about a minute; writes data/ (75 MB)
  ```

  The access keys expire about an hour after they are issued, so build right after downloading them. The script
  samples with fixed seeds, so everyone gets the identical sample.

## Editing the notebook

Edit `build_notebook.py`, then run `python build_notebook.py` (needs `nbformat`). That writes a notebook without
outputs, so run it top to bottom again before committing.
