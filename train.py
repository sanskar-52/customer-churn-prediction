import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def load_data():
    # Dataset URL provided in project instructions
    url = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # Convert TotalCharges to numeric and drop missing rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Drop customerID column (not useful for ML)
    df = df.drop("customerID", axis=1)

    # Encode all categorical columns using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Target variable: Churn_Yes (1 means churn, 0 means not churn)
    y = df["Churn_Yes"]
    X = df.drop("Churn_Yes", axis=1)

    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    X_train, X_test, y_train, y_test = load_data()

    # Basic logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict & evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save output for validator
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": float(accuracy)}, f, indent=2)

    print(f"Training complete! Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train()