from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import os
import pandas as pd

# Config - update paths if needed
MODEL_PATH = os.getenv("CALORIE_MODEL_PATH",
                       r"/home/deepak/Downloads/fitness calorie (3)(1)/fitness calorie (2)/fitness calorie/random_forest_regressor.joblib")

# Load model once
model = load(MODEL_PATH)

app = Flask(__name__)
CORS(app)


def predict_calories(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp):
    input_df = pd.DataFrame(
        [[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]],
        columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"],
    )
    pred = model.predict(input_df)
    return float(pred[0])


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Request JSON example:
    {
      "Gender": 1,
      "Age": 30,
      "Height": 175.0,
      "Weight": 70.0,
      "Duration": 45,
      "Heart_Rate": 140,
      "Body_Temp": 36.6
    }
    """
    data = request.get_json(force=True)
    required = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        pred = predict_calories(
            int(data["Gender"]),
            float(data["Age"]),
            float(data["Height"]),
            float(data["Weight"]),
            float(data["Duration"]),
            float(data["Heart_Rate"]),
            float(data["Body_Temp"]),
        )
    except Exception as e:
        return jsonify({"error": "Invalid input or model error", "detail": str(e)}), 400

    return jsonify({"predicted_calories": pred})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
