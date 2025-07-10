import uvicorn as uv
from fastapi import FastAPI
from pydantic import BaseModel
from pyexpat import features

from data.loadCsv import CSVLoader
from model.naive_bayesian import NaiveBayes
from prediction.checking import Prediction
from prediction.evalution import ModelTester

app = FastAPI()

data = CSVLoader("C:/Users/User/Downloads/buy_computer_data.csv")
df = data.load()
model = NaiveBayes(df, 'buys_computer')
model.fit()
predictor = Prediction(model)
tester = ModelTester(model)

@app.post("/predict")
def predict(row: dict[str,str]):
    try:
        pred = predictor.prediction(row)
        return {"prediction": pred}
    except Exception as e:
        return {'error': str(e)}

@app.get('/test/full')
def test_full_accuracy():
    acc = tester.test_full_dataset_prediction()
    return {'accuracy': acc}

@app.get('/test/split')
def test_split_accuracy():
    acc = tester.test_with_train_test_split()
    return {'accuracy': acc}

@app.get("/features")
def get_features():
    features_options = {}

    for feature in model.features:
        features_options[feature] = list(df[feature].unique())

    return {'features':features_options}

if __name__ == '__main__':
    uv.run('server:app', host='127.0.0.1', port=8000, reload=True)