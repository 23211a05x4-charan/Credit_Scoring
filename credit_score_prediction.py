import pandas as pd

# Load dataset
data = pd.read_csv("test.csv")

# Show first rows
print(data.head())

# Show column names
print(data.columns)
print("Dataset Shape:", data.shape)
# Remove unnecessary columns
data = data.drop(["ID", "Customer_ID", "Name", "SSN"], axis=1)

# Check remaining columns
print("After removing unwanted columns:")
print(data.columns)

print("New Shape:", data.shape)
from sklearn.preprocessing import LabelEncoder

for col in data.columns:
    print("Checking:", col, data[col].dtype)

    if col != "Credit_Mix":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

print(data.dtypes)


target_encoder = LabelEncoder()
data["Credit_Mix"] = target_encoder.fit_transform(data["Credit_Mix"])
y = data["Credit_Mix"]
X = data.drop("Credit_Mix", axis=1)
from sklearn.model_selection import train_test_split
print(X.dtypes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model Trained Successfully")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")
sample = X_test.iloc[[0]]      # take one sample row

prediction = model.predict(sample)

actual_label = target_encoder.inverse_transform(prediction)

print("Predicted Credit Mix:", actual_label[0])