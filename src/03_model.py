"""Small CNN encoder + LSTM + decoder, for a first working baseline."""
import torch
from torch import nn


class RadarCNNLSTM(nn.Module):
    def __init__(self, image_size=64, hidden_size=128):
        super().__init__()
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        encoded_size = 16 * (image_size // 4) ** 2
        self.lstm = nn.LSTM(encoded_size, hidden_size, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, image_size * image_size), nn.Sigmoid()
        )

    def forward(self, frames):
        batch, steps, height, width = frames.shape
        encoded = self.encoder(frames.reshape(batch * steps, 1, height, width))
        encoded = encoded.reshape(batch, steps, -1)
        output, _ = self.lstm(encoded)
        return self.decoder(output[:, -1]).reshape(batch, height, width)
