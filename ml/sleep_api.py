from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Load trained model and label encoders
model = joblib.load("sleep_recommender.pkl")
label_encoders = joblib.load("sleep_label_encoders.pkl")

# ✅ Load dataset columns to ensure correct output mapping
df = pd.read_csv("Updated_Sleep_Dataset.csv")

# ✅ Define input & output features
input_features = ["SLEEP TIME", "SLEEP INTERRUPTION"]
output_features = [col for col in df.columns if col not in input_features]

@app.route("/api/sleep", methods=["POST"])
def recommend_sleep():
    data = request.json
    print("➡ Received Data:", data)

    try:
        # ✅ Extract input values
        sleep_time = float(data.get("SLEEP TIME"))
        sleep_interruption = float(data.get("SLEEP INTERRUPTION"))

        # ✅ Prepare input array
        input_data = np.array([[sleep_time, sleep_interruption]])

        # ✅ Predict output
        predicted_output = model.predict(input_data)[0]  # Ensure correct indexing

        # ✅ Ensure correct indexing of output values
        if len(predicted_output) != len(output_features):
            return jsonify({"error": f"Model output size mismatch: Expected {len(output_features)}, got {len(predicted_output)}"}), 500

        # ✅ Convert categorical outputs back to labels
        decoded_output = {}

        for idx, column in enumerate(output_features):
            if column in label_encoders:
                try:
                    decoded_output[column] = label_encoders[column].inverse_transform([int(predicted_output[idx])])[0]
                except ValueError:
                    decoded_output[column] = "Unknown"  # Handle unseen labels
            else:
                decoded_output[column] = float(predicted_output[idx])  # Keep numeric values as is

        # ✅ Return JSON response
        return jsonify(decoded_output)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
