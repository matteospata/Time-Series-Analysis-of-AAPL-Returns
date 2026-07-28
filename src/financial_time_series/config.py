from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    seed: int = _int_env("FINANCIAL_SEED", 42)
    lstm_epochs: int = _int_env("FINANCIAL_LSTM_EPOCHS", 30)
    lstm_window: int = _int_env("FINANCIAL_LSTM_WINDOW", 20)


settings = Settings()

