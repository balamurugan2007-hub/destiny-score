import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'sleep_hours': np.random.uniform(4, 9, n),
    'exercise_days': np.random.randint(0, 7, n),
    'junk_food_days': np.random.randint(0, 7, n),
    'screen_time': np.random.uniform(1, 12, n),
    'social_media_hours': np.random.uniform(0, 6, n),
    'study_hours': np.random.uniform(0, 8, n),
    'stress_level': np.random.randint(1, 10, n),
    'savings_percent': np.random.uniform(0, 50, n),
    'reading_hours': np.random.uniform(0, 3, n),
    'water_intake': np.random.uniform(1, 4, n),
})

# Destiny Score calculate பண்றோம் (0-100)
data['destiny_score'] = (
    (data['sleep_hours'] / 9) * 20 +
    (data['exercise_days'] / 7) * 20 +
    (1 - data['junk_food_days'] / 7) * 10 +
    (1 - data['stress_level'] / 10) * 15 +
    (data['study_hours'] / 8) * 15 +
    (data['savings_percent'] / 50) * 10 +
    (data['reading_hours'] / 3) * 5 +
    (data['water_intake'] / 4) * 5
).clip(0, 100)

data.to_csv('data/lifestyle_data.csv', index=False)
print("✅ Dataset created successfully!")
print(data.head())