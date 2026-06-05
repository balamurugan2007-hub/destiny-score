import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Data load பண்றோம்
data = pd.read_csv('data/lifestyle_data.csv')

X = data.drop('destiny_score', axis=1)
y = data['destiny_score']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale பண்றோம்
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model train பண்றோம்
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Accuracy check பண்றோம்
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📊 MAE  : {mae:.2f}")
print(f"📊 R2 Score: {r2:.4f}")

# Model save பண்றோம்
pickle.dump(model, open('model/destiny_model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
print("✅ Model saved in /model folder!")