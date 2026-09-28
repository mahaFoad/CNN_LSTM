"""Load the saved model and forecast from the last prepared sequence."""
from pathlib import Path
import importlib.util

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("radar_model", Path(__file__).with_name("03_model.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    checkpoint_path = ROOT / "models/radar_cnn_lstm.pt"
    data_path = ROOT / "data/processed/sequences/radar_sequences.npz"
    if not checkpoint_path.exists() or not data_path.exists():
        raise SystemExit("Run preparation and training before prediction.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = module.RadarCNNLSTM(checkpoint["image_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with np.load(data_path) as prepared:
        last_input = torch.from_numpy(prepared["X"][-1:].copy())
    with torch.no_grad():
        prediction = model(last_input)[0].numpy()
    output = ROOT / "outputs/next_radar_frame.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prediction * 255).clip(0, 255).astype(np.uint8)).save(output)
    print(f"Saved predicted grayscale frame: {output}")


if __name__ == "__main__":
    main()
