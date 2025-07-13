import uvicorn as uv
from fastapi import FastAPI
from data.loadCsv import CSVLoader
from data.clean_data import Cleaner
from model.naive_bayesian import NaiveBayes
from prediction.checking import Prediction
from prediction.evalution import ModelTester

app = FastAPI()

data = CSVLoader("C:/Users/User/Downloads/buy_computer_data.csv")
df = data.load()
df = Cleaner(df).clean_data()
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
        return {'error': str(e)}

@app.get('/test/full')
def test_full_accuracy():
    acc = tester.test_full_dataset_prediction()
    return {'accuracy': acc}

@app.get("/test/split")
def test_split_accuracy():
    try:
        acc = tester.test_with_train_test_split()
        return {"accuracy": acc}
    except Exception as e:
        print(f"[ERROR] /test/split: {e}")
        return {"error": str(e)}


@app.get("/features")
def get_features():
    features_options = {}

    for feature in model.features:
        features_options[feature] = list(df[feature].unique())

    return {'features':features_options}

if __name__ == '__main__':
    uv.run('server:app', host='127.0.0.1', port=8000, reload=True)

