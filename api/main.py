
from fastapi import FastAPI
import random

app = FastAPI(title="Supply Chain Forecast API")

@app.get("/")
def home():
    return {"message": "Demand Forecasting API Running"}

@app.get("/forecast")
def forecast():
    result = {
        "forecast": [random.randint(100, 300) for _ in range(7)]
    }
    return result
