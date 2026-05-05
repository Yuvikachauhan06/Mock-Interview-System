import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("training_data.csv")

df.columns = df.columns.str.strip()

X = df[[
    "similarity_score",
    "keyword_match_score",
    "answer_length",
    "filler_count",
    "technical_depth_score"
]].values

label_map = {"Poor": 0, "Average": 1, "Good": 2}
reverse_label_map = {0: "Poor", 1: "Average", 2: "Good"}

y = np.array([label_map[val] for val in df["label"].values])

mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-9
X_scaled = (X - mean) / std

def train_test_split_custom(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    split = int((1 - test_size) * len(X))
    return X[:split], X[split:], y[:split], y[split:]

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split_custom(X_scaled, y, test_size)

#  LOGISTIC REGRESSION  
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=2000):
        self.lr = lr
        self.epochs = epochs

    def softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1

        for epoch in range(self.epochs):
            linear = np.dot(X, self.W) + self.b
            probs = self.softmax(linear)

            dW = (1 / n_samples) * np.dot(X.T, (probs - y_onehot))
            db = (1 / n_samples) * np.sum(probs - y_onehot, axis=0, keepdims=True)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 500 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        linear = np.dot(X, self.W) + self.b
        return self.softmax(linear)

print("Training Logistic Regression...")
model = LogisticRegression(lr=0.01, epochs=2000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)
y_pred = np.argmax(probs, axis=1)
accuracy = np.mean(y_pred == y_test)

print("\n RESULTS")
print("=" * 40)
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(mean, open("mean.pkl", "wb"))
pickle.dump(std, open("std.pkl", "wb"))

print("\n model saved successfully!")