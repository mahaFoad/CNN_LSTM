# Radar CNN–LSTM project

Open this folder in VS Code (`File > Open Folder`). The first goal is to put radar files in the correct place and inspect their format. The scripts assume **PNG/JPEG images in ZIP files** and a **5-minute interval**. If your download is HDF5, NetCDF, or another radar format, its decoding needs a different extraction step; do not rename it to `.zip`.

## Folders

```text
radar_cnn_lstm/
├── data/
│   ├── raw/
│   │   ├── radar_archives/   # downloaded ZIP files
│   │   └── numerical/        # rain gauges / other numerical data (future use)
│   └── processed/
│       ├── radar_images/     # extracted PNG/JPEG files
│       └── sequences/        # prepared arrays
├── models/                    # trained .pt checkpoints
├── outputs/                   # plots and predictions later
├── src/
│   ├── 01_extract_radar.py
│   ├── 02_prepare_data.py
│   ├── 03_model.py
│   ├── 04_train.py
│   └── 05_predict.py
├── requirements.txt
└── .gitignore
```

## Set up Python in VS Code

In the VS Code terminal, from this folder:

```bash
python -m venv .venv
```

Activate it on Windows with `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`). Select `.venv` using **Python: Select Interpreter**, then:

```bash
python -m pip install -r requirements.txt
```

## Run in order

1. Download radar ZIP files into `data/raw/radar_archives/`. Keep original downloads. Put numerical CSV files in `data/raw/numerical/` when available. Those CSV files are reserved for a later extension and are **not** used by this image-only baseline.
2. `python src/01_extract_radar.py` extracts images and skips duplicate file contents. The extractor never deletes original ZIPs.
3. Check the extracted files. Their names must contain a UTC timestamp like `20260124_1000.png` (also accepts `202601241000` or `2026-01-24_10-00`). If the supplier uses other names, adapt the timestamp parser before step 4. Images need the same map extent and colour scale.
4. `python src/02_prepare_data.py` makes six consecutive input frames and the following frame as target. It skips gaps and writes `data/processed/sequences/radar_sequences.npz`.
5. `python src/04_train.py` trains a small CNN–LSTM baseline and saves `models/radar_cnn_lstm.pt` (including weights and settings). It uses CPU automatically if CUDA is unavailable.
6. `python src/05_predict.py` loads the saved checkpoint and writes one forecast image into `outputs/`.

**What this predicts:** the next radar image at +5 minutes, using the previous 30 minutes of radar images. This starter model treats pixels as grayscale intensity. It is **not calibrated rainfall in mm/h**: radar colours and metadata need decoding before scientifically interpreting or scoring rainfall. We can adapt extraction and targets after inspecting one real downloaded file.
