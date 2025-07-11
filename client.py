import requests


class Controller:
    def __init__(self):
        self.url = "http://127.0.0.1:8000"
        self.features_info = self.fetch_features_from_server()

    def fetch_features_from_server(self):
        res = requests.get(f"{self.url}/features")
        if res.ok:
            return res.json()["features"]
        else:
            raise Exception("Error receiving properties from the server")

    def run(self):
        while True:
            print("\nבחר פעולה:")
            print("1. חיזוי על כל הדאטה")
            print("2. חיזוי עם Train/Test split")
            print('3. חיזוי על שורה שהוזנה ע"י המשתמש')
            print("4. יציאה")
            choice = input(">> ")

            if choice == "1":
                self.test_full()
            elif choice == "2":
                self.test_split()
            elif choice == "3":
                self.predict_single_row()
            elif choice == "4":
                print("להתראות!")
                break
            else:
                print("בחירה לא חוקית.")

    def test_full(self):
        res = requests.get(f"{self.url}/test/full")
        if res.ok:
            try:
                accuracy = res.json().get('accuracy')
                print(f"\n✅ accurate: {accuracy:.2%}")
            except Exception as e:
                print(f"❌ שגיאת JSON: {e}")
        else:
            print(f"❌ שגיאה בבקשה: {res.status_code} {res.reason}")

    def test_split(self):
        res = requests.get(f"{self.url}/test/split")
        acc = res.json().get('accuracy')
        print(
            f"✅ Accuracy on test set ({int((1 - acc[1]) * 100)}% train / {int(acc[1] * 100)}% test): {acc[0]:.2%}")

    def predict_single_row(self):
        print("\nהכנס ערכים עבור כל תכונה:")
        row = {}
        for feature,allowed_vals in self.features_info.items():
            allowed_str = ", ".join(allowed_vals)

            while True:
                val = input(f"{feature} options: ({allowed_str}) ").strip()
                if val in allowed_vals:
                    row[feature] = val
                    break
                else:
                    print(f"❌ ערך לא חוקי! נסה שוב. האפשרויות הן: {allowed_str}")

        res = requests.post("http://127.0.0.1:8000/predict", json=row)
        if res.ok:
            print(f"\n✅ התחזית: {res.json()['prediction']}")
        else:
            print("❌ שגיאה בשליחת בקשה לשרת")



if __name__ == '__main__':
    Controller().run()
