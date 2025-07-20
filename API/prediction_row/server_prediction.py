import uvicorn as uv
from fastapi import FastAPI
from prediction.checking import Prediction
import pickle
from logger.Logger import logger

app = FastAPI()

with open("../saved_model.pkl", "rb") as f:
    model = pickle.load(f)

predictor = Prediction(model)

@app.post("/predict")
def predict(row: dict):
    try:
        pred = predictor.prediction(row)
        return {"prediction": pred}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {'error': str(e)}

if __name__ == '__main__':
    logger.info("🚀 Starting FastAPI server on http://127.0.0.1:8001")
    uv.run('server_prediction:app', host='127.0.0.1', port=8001, reload=True)