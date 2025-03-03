import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ✅ Load dataset
df = pd.read_csv("medicine_data.csv")

# ✅ Convert categorical columns to lowercase and strip spaces
df = df.applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# ✅ Fill missing values
df.fillna("unknown", inplace=True)

# ✅ Encode Disease as a Label
le_disease = LabelEncoder()
df["Disease"] = le_disease.fit_transform(df["Disease"])

# ✅ Store mappings for later use
label_encoders = {"Disease": le_disease}

# ✅ Define Features (X) and Target (y)
X = df[["Disease"]]
y = df.drop(columns=["Disease"])  # Output should include all other details

# ✅ Convert categorical outputs to strings (to prevent NaN)
y = y.astype(str)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Reduce Model Complexity (Prevent MemoryError)
model = RandomForestClassifier(
    n_estimators=20,  # ✅ Reduce trees from 200 to 20
    max_depth=6,  # ✅ Reduce tree depth
    max_features="sqrt",  # ✅ Use fewer features per split
    max_samples=0.5,  # ✅ Train on 50% of data per tree
    random_state=42
)

# ✅ Train the Model
model.fit(X_train, y_train)

# ✅ Save model & encoders without compression
joblib.dump(model, "medicine_model.joblib")  # ✅ Remove compression to avoid `mmap_mode` error
joblib.dump(label_encoders, "medicine_encoders.joblib")

print("✅ Medicine Recommendation ML Model Trained and Saved!")
