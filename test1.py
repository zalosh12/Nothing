import requests

# כתובות השרתים (כמו ב-Docker, או localhost עם פורטים)
TRAIN_SERVER_URL = "http://localhost:8005"
PREDICT_SERVER_URL = "http://localhost:8005"

def train_model(data_url):
    print("Sending training request...")
    res = requests.post(f"{TRAIN_SERVER_URL}/train/", json={"url": data_url})
    if res.status_code == 200:
        print("Training completed successfully!")
        print("Accuracy:", res.json().get("accuracy"))
    else:
        print("Training failed:", res.text)

def get_model_from_train_server():
    print("Getting model from train server...")
    res = requests.get(f"{TRAIN_SERVER_URL}/export_model/")
    if res.status_code == 200:
        model_dict = res.json()
        print("Model fetched successfully!")
        return model_dict
    else:
        print("Failed to get model:", res.text)
        return None

def send_model_to_predict_server(model_dict):
    print("Sending model to predict server...")
    res = requests.post(f"{PREDICT_SERVER_URL}/load_model/", json={"model_dict": model_dict})
    if res.status_code == 200:
        print("Model loaded successfully on predict server!")
    else:
        print("Failed to load model on predict server:", res.text)

def predict_sample(features):
    print("Sending predict request...")
    res = requests.post(f"{PREDICT_SERVER_URL}/predict/", json={"features": features})
    if res.status_code == 200:
        print("Prediction result:", res.json().get("prediction"))
    else:
        print("Prediction failed:", res.text)

if __name__ == "__main__":
    # שלב 1: מאמן מודל על קובץ דאטה (קישור URL)
    data_url = "https://raw.githubusercontent.com/zalosh12/la_liga_csv/refs/heads/main/car_evaluation"
    train_model(data_url)

    # שלב 2: מביא את המודל מהשרת אימון
    model = get_model_from_train_server()
    if model:
        # שלב 3: שולח את המודל לשרת הפרדיקט
        send_model_to_predict_server(model)

        # שלב 4: מבצע חיזוי לדוגמה
        sample_features = ["feature_value1", "feature_value2", "feature_value3"]  # תחליף לפי המודל שלך
        predict_sample(sample_features)
