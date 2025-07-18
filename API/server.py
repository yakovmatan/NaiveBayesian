import uvicorn as uv
from fastapi import FastAPI
from data.loadCsv import CSVLoader
from data.clean_data import Cleaner
from model.naive_bayesian import NaiveBayes
from prediction.checking import Prediction
from prediction.evalution import ModelTester
from logger.Logger import logger

app = FastAPI()

logger.info("Loading and cleaning data...")
data = CSVLoader("C:/Users/User/Downloads/buy_computer_data.csv")
df = data.load()
df = Cleaner(df).clean_data()


logger.info("Initializing and training Naive Bayes model...")
model = NaiveBayes(df, 'buys_computer')
model.fit()

predictor = Prediction(model)
tester = ModelTester(model,predictor)

@app.post("/predict")
def predict(row: dict):
    try:
        pred = predictor.prediction(row)
        return {"prediction": pred}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {'error': str(e)}

@app.get('/test/full')
def test_full_accuracy():
    try:
        acc = tester.test_full_dataset_prediction()
        return {'accuracy': acc}
    except Exception as e:
        logger.error(f"Full dataset test error: {str(e)}")
        return {'error': str(e)}

@app.get("/test/split")
def test_split_accuracy():
    try:
        acc = tester.test_with_train_test_split()
        return {"accuracy": acc}
    except Exception as e:
        logger.error("Train test split testing error: %s", str(e))
        return {"error": str(e)}


@app.get("/features")
def get_features():
    try:
        logger.info("🔎 Fetching model features")
        features_options = {}
        for feature in model.features:
            features_options[feature] = list(df[feature].unique())
        logger.info("📋 Features and options fetched successfully")
        return {'features': features_options}
    except Exception as e:
        logger.error("❌ Error fetching feature options: %s", str(e))
        return {'error': str(e)}

if __name__ == '__main__':
    logger.info("🚀 Starting FastAPI server on http://127.0.0.1:8000")
    uv.run('server:app', host='127.0.0.1', port=8000, reload=True)

