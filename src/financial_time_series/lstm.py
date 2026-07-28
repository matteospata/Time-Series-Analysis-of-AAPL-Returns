from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 32, layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.head(output[:, -1, :])


@dataclass
class LSTMResult:
    model: LSTMRegressor
    window: int
    mean: float
    standard_deviation: float
    history: list[float]

    def forecast(self, history: pd.Series | np.ndarray, horizon: int) -> np.ndarray:
        values = np.asarray(history, dtype=float)
        if len(values) < self.window:
            raise ValueError("The history must contain at least as many observations as the LSTM window.")
        scaled = (values - self.mean) / self.standard_deviation
        sequence = list(scaled[-self.window:])
        predictions: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                tensor = torch.tensor(sequence[-self.window:], dtype=torch.float32).reshape(1, self.window, 1)
                prediction = float(self.model(tensor).item())
                predictions.append(prediction * self.standard_deviation + self.mean)
                sequence.append(prediction)
        return np.asarray(predictions)


def fit_lstm(values: pd.Series, window: int = 20, epochs: int = 30, hidden_size: int = 32, learning_rate: float = 1e-3, seed: int = 42) -> LSTMResult:
    seed_everything(seed)
    observations = np.asarray(values.dropna(), dtype=float)
    if len(observations) <= window + 2:
        raise ValueError("The LSTM requires more observations than window + 2.")
    mean = float(observations.mean())
    standard_deviation = float(observations.std()) or 1.0
    scaled = (observations - mean) / standard_deviation
    features = np.stack([scaled[index - window:index] for index in range(window, len(scaled))])
    targets = scaled[window:]
    x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    model = LSTMRegressor(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        prediction = model(x_tensor)
        loss = criterion(prediction, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return LSTMResult(model, window, mean, standard_deviation, history)

