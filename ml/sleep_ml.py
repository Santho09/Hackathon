import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ✅ Load Dataset
file_path = "Updated_Sleep_Dataset.csv"
df = pd.read_csv(file_path)

# ✅ Display dataset columns
print("📌 Dataset Columns:", df.columns)

# ✅ Select Features (Input) and Target (Output)
input_features = ["SLEEP TIME", "SLEEP INTERRUPTION"]  # Features to be given as input
target_features = [col for col in df.columns if col not in input_features]  # Predict all other columns

# ✅ Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# ✅ Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# ✅ Prepare Data
X = df[input_features]  # Inputs
y = df[target_features]  # Outputs

# ✅ Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save Model & Encoders
joblib.dump(model, "sleep_recommender.pkl")
joblib.dump(label_encoders, "sleep_label_encoders.pkl")

# ✅ Test the Model
sample_input = np.array([[7, 2]])  # Example: 7 hours of sleep, 2 interruptions
predicted_output = model.predict(sample_input)

print("\n✅ Sample Input:", sample_input)
print("✅ Predicted Output:", predicted_output)

print("\n🎉 Sleep Recommender Model Trained & Saved Successfully!")
