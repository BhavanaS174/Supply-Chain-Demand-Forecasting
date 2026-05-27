
import pandas as pd
import numpy as np

print("Running demand forecast...")

forecast = pd.DataFrame({
    "Day": np.arange(1, 8),
    "Forecasted_Demand": np.random.randint(100, 300, 7)
})

print(forecast)
