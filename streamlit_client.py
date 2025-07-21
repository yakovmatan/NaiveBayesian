import streamlit as st
import requests

URL_FEATURES = "http://127.0.0.1:8000/features"
URL_TEST_FULL = "http://127.0.0.1:8000/test/full"
URL_TEST_SPLIT = "http://127.0.0.1:8000/test/split"
URL_PREDICT = "http://127.0.0.1:8001/predict"


@st.cache_data
def get_features_info():
    res = requests.get(URL_FEATURES)
    if res.ok:
        return res.json()["features"]
    else:
        st.error("❌ Error retrieving features from the server.")
        return {}


def test_full():
    res = requests.get(URL_TEST_FULL)
    if res.ok:
        accuracy = res.json().get('accuracy', None)
        if accuracy is not None:
            st.success(f"✅ Accuracy on full dataset: {accuracy:.2%}")
        else:
            st.error("❌ Response did not include accuracy value.")
    else:
        st.error(f"❌ Server error: {res.status_code} {res.reason}")


def test_split():
    res = requests.get(URL_TEST_SPLIT)
    if res.ok:
        acc = res.json().get('accuracy', None)
        if acc:
            train_percent = int((1 - acc[1]) * 100)
            test_percent = int(acc[1] * 100)
            st.success(f"✅ Accuracy on test set ({train_percent}% train / {test_percent}% test): {acc[0]:.2%}")
        else:
            st.error("❌ Invalid server response.")
    else:
        st.error("❌ Failed to contact server.")


def predict_single_row(features_info):
    st.subheader("Enter values for each feature:")
    user_input = {}

    for feature, allowed_vals in features_info.items():
        user_input[feature] = st.selectbox(f"{feature}", allowed_vals)

    if st.button("Predict"):
        res = requests.post(URL_PREDICT, json=user_input)
        if res.ok:
            prediction = res.json().get("prediction")
            st.success(f"✅ Prediction result: {prediction}")
        else:
            st.error("❌ Failed to send request to the prediction server.")



st.set_page_config(page_title="Prediction App", layout="centered", page_icon="🤖")

st.title("🤖 Machine Learning Prediction App")

features_info = get_features_info()

option = st.radio(
    "Choose an action:",
    ["Predict on full dataset", "Train/Test split prediction", "Manual row prediction"]
)

if option == "Predict on full dataset":
    test_full()

elif option == "Train/Test split prediction":
    test_split()

elif option == "Manual row prediction":
    if features_info:
        predict_single_row(features_info)
    else:
        st.warning("No feature information retrieved from server.")
