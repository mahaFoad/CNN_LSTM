"""Train on past radar frames and save the best checkpoint."""
from pathlib import Path
import importlib.util

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/sequences/radar_sequences.npz"
SAVE = ROOT / "models/radar_cnn_lstm.pt"
EPOCHS = 10
BATCH_SIZE = 16

# The numbered filename is readable in VS Code; import it explicitly here.
spec = importlib.util.spec_from_file_location("radar_model", Path(__file__).with_name("03_model.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RadarCNNLSTM = module.RadarCNNLSTM


def main():
    if not DATA.exists():
        raise SystemExit("Prepared data missing. Run 01_extract_radar.py and 02_prepare_data.py first.")
    with np.load(DATA) as prepared:
        X = torch.from_numpy(prepared["X"].copy())
        y = torch.from_numpy(prepared["y"].copy())
    if len(X) < 10:
        raise SystemExit("Need at least 10 sequences for this first training run.")

    # Chronological split: later frames are kept for validation.
    split = int(len(X) * 0.8)
    train = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=BATCH_SIZE, shuffle=True)
    validation = DataLoader(TensorDataset(X[split:], y[split:]), batch_size=BATCH_SIZE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RadarCNNLSTM(image_size=X.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    SAVE.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for inputs, targets in validation:
                prediction = model(inputs.to(device))
                losses.append(loss_fn(prediction, targets.to(device)).item())
        val_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1}/{EPOCHS}: validation MSE = {val_loss:.5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": model.state_dict(), "image_size": X.shape[-1],
                        "frames": X.shape[1], "best_val_mse": best_loss}, SAVE)

    print(f"Best trained model saved: {SAVE}")


if __name__ == "__main__":
    main()
