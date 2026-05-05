from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel


with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class BitcoinData(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float

@app.get("/")
def home():
    return {"message": "Bitcoin Prediction API is running"}

@app.post("/predict")
def predict(data: BitcoinData):

    df = pd.DataFrame([data.dict()])
    

    prediction = model.predict(df)
    
    return {"prediction_next_minute_close": float(prediction[0])}
