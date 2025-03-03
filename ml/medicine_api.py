from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load trained model (without `mmap_mode="r"`)
model = joblib.load("medicine_model.joblib")

# ✅ Load label encoders
encoders = joblib.load("medicine_encoders.joblib")

@app.route("/api/medicine", methods=["POST"])
def recommend_medicine():
    try:
        data = request.json
        print("✅ Received Data:", data)

        # ✅ Encode disease
        disease = data["Disease"].strip().lower()
        if disease not in encoders["Disease"].classes_:
            return jsonify({"error": f"Invalid disease: {disease}"}), 400

        encoded_disease = encoders["Disease"].transform([disease])[0]

        # ✅ Predict medicine details
        medicine_details = model.predict(np.array([[encoded_disease]]))[0]

        response = {
            "Medicine Name": medicine_details[0],
            "Composition": medicine_details[1],
            "Side Effects": medicine_details[2],
            "Excellent Review %": medicine_details[3],
            "Average Review %": medicine_details[4],
            "Poor Review %": medicine_details[5]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5004, debug=True)
