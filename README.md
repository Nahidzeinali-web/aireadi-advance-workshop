# AI-READI ADVANCE workshop

A hands-on notebook for AI-READI workshop. It uses a small sample of the ADVANCE Center synthetic AI-READI dataset and runs offline on a laptop
processor; no cloud account is needed.

## Before the session

1. **Download this repository.** On its GitHub page choose **Code**, then **Download ZIP**, and unzip it (or
   `git clone` it).
2. **Add the data.** The workshop organizers send the data separately, as `aireadi_workshop_data.zip`. Unzip it inside
   the downloaded folder so that a folder named `data` sits next to `aireadi_synthetic_demo.ipynb`, with `scans/`,
   `gallery/`, `scans.csv`, `gallery.csv` and `participants.csv` directly inside it. If your computer unzips it to a
   folder with another name (such as `data 2` or `aireadi_workshop_data`), that works too.
3. **Install and test.** Follow [Setup](#setup) below: open the notebook and run the first code cell, which installs
   everything else. Then run the next cell; it should report 2,000 scans.

If your computer will not cooperate, there is a backup: the organizers can give you a web link and password for a
ready-made copy that runs in the browser, with nothing to install.

| File | What it is |
|---|---|
| `aireadi_synthetic_demo.ipynb` | the notebook, saved with every cell's output |
| `PRESENTER_NOTES.md` | talking points and rough timings for each cell; cell numbers match the notebook |
| `build_notebook.py` | source for both files above |
| `data/README.md`, `data/LICENSE-AI-READI-synthetic-data.pdf` | what the sample contains, and the license that governs it |
| `tools/build_sample_data.py` | builds the `data` folder from the dataset with your own access key |
| `hosting/` | the browser-based backup: one shared password, a separate copy for each person (see `hosting/README.md`) |

## Computer requirements

| | Needed | Comfortable |
|---|---|---|
| Processor | 2 cores | 4 or more cores |
| Memory | 4 GB | 8 GB |
| Free disk space | 2 GB (software about 1.5 GB, data 75 MB) | |
| Software | Python 3.10 or newer, and Jupyter (JupyterLab, Jupyter Notebook or VS Code) | |
| Internet | once, for the install cell | |
| Time to run every cell | about 5 minutes on 4 cores | |

No graphics card is needed.

## Setup

Each participant runs the notebook on their own computer.

1. **Python and Jupyter.** If you already use Jupyter (for example through Anaconda or VS Code), skip to step 2.
   Otherwise install Python 3.10 or newer from python.org, then in a terminal:

   ```bash
   python -m venv aireadi-env
   source aireadi-env/bin/activate            # Windows: aireadi-env\Scripts\activate
   pip install jupyterlab
   ```

2. **Files.** Put `aireadi_synthetic_demo.ipynb` in a folder, and next to it a folder named `data` holding the
   workshop files (`scans/`, `gallery/`, `scans.csv`, `gallery.csv`, `participants.csv`). Unzipping
   `aireadi_workshop_data.zip` in the notebook's folder creates exactly this.

3. **Open and install.** Start Jupyter (`jupyter lab`, from the activated environment), open the notebook, and run
   the first code cell. It installs everything else the notebook needs into the Python running the notebook,
   skipping anything already there. The first time it downloads about 300 MB.

If the install cell reports an "externally managed environment", your computer's own Python does not allow
installs; use the environment from step 1. `requirements.txt` lists the same packages for installing by hand.

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
