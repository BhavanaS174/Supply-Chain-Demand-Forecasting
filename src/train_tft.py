import os

os.makedirs("models", exist_ok=True)

import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer

print("Loading dataset...")

# Dummy dataset
data = pd.DataFrame({
    "time_idx": np.tile(np.arange(100), 5),
    "sku": np.repeat([f"SKU_{i}" for i in range(5)], 100),
    "sales": np.random.randint(10, 200, 500),
    "promotion": np.random.randint(0, 2, 500)
})

training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="sales",
    group_ids=["sku"],
    max_encoder_length=24,
    max_prediction_length=12,
    time_varying_known_reals=["time_idx", "promotion"],
    time_varying_unknown_reals=["sales"],
    target_normalizer=GroupNormalizer(groups=["sku"])
)

train_dataloader = training.to_dataloader(train=True, batch_size=64)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss()
)

trainer = Trainer(max_epochs=3)

trainer.fit(tft, train_dataloader)

import os
os.makedirs("models", exist_ok=True)

torch.save(tft.state_dict(), "models/tft_model.pth")

print("Model training completed.")