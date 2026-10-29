import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load dataset
iris = load_iris()
X = iris.data
Y = iris.target

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Predictions
Y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

#confusion matrix
cm = confusion_matrix(y_test, Y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.xlabel("Predict")
plt.ylabel("Actual")
plt.title("ConfusionMatrix")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Save model
joblib.dump(model, "outputs/model.joblib")

print("Model and confusion matrix saved in outputs/")