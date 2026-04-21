import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("training_data.csv")

# Features
X = df[[
    "similarity_score",
    "keyword_match_score",
    "answer_length",
    "filler_count",
    "technical_depth_score"
]]

# Labels
y = df["label"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved!")