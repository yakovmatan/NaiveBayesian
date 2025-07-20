from API.training_model.server_model import predictor
from data.loadCsv import CSVLoader
from model.naive_bayesian import NaiveBayes
from prediction.checking import Prediction
from prediction.evalution import ModelTester


class Controller:
    def __init__(self):
        self.data = CSVLoader("C:/Users/User/Downloads/buy_computer_data.csv")
        self.df = self.data.load()
        self.model = NaiveBayes(self.df, 'buys_computer')
        self.model.fit()
        self.predictor = Prediction(self.model)
        self.tester = ModelTester(self.model,predictor)




    def run(self):
        while True:
            print("\nבחר פעולה:")
            print("1. חיזוי על כל הדאטה")
            print("2. חיזוי עם Train/Test split")
            print('3. חיזוי על שורה שהוזנה ע"י המשתמש')
            print("4. יציאה")
            choice = input(">> ")

            if choice == "1":
                self.tester.test_full_dataset_prediction()
            elif choice == "2":
                self.tester.test_with_train_test_split()
            elif choice == "3":
                self.predict_single_row()
            elif choice == "4":
                print("להתראות!")
                break
            else:
                print("בחירה לא חוקית.")

    def predict_single_row(self):
        print("\nהכנס ערכים עבור כל תכונה:")
        row = {}
        for feature in self.model.features:
            allowed_vals = self.df[feature].unique()
            allowed_str = ", ".join(allowed_vals)

            while True:
                val = input(f"{feature} options: ({allowed_str}) ").strip()
                if val in allowed_vals:
                    row[feature] = val
                    break
                else:
                    print(f"❌ ערך לא חוקי! נסה שוב. האפשרויות הן: {allowed_str}")

        prediction = self.predictor.prediction(row)
        print(f"\n✅ התחזית: {prediction}")



if __name__ == '__main__':
    Controller().run()