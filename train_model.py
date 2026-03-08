import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Selected features
X = df[["mean radius", "mean texture", "mean perimeter", "mean area"]]
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save accuracy
with open("accuracy.pkl", "wb") as f:
    pickle.dump(accuracy, f)

print("Model trained successfully!")
print("Accuracy:", accuracy)
