import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load data
shipments = pd.read_csv("data/shipments.csv")

# 2. Convert dates
date_columns = [
    "order_date",
    "promised_delivery_date",
    "actual_delivery_date"
]

for col in date_columns:
    shipments[col] = pd.to_datetime(shipments[col], errors="coerce")

# 3. Data validation
shipments = shipments.dropna(subset=date_columns)
shipments = shipments.drop_duplicates(subset=["shipment_id"])
shipments = shipments[shipments["weight_kg"] > 0]

# 4. Create target
shipments["is_late"] = (
    shipments["actual_delivery_date"] >
    shipments["promised_delivery_date"]
).astype(int)

# 5. Feature engineering
shipments["order_day"] = shipments["order_date"].dt.day
shipments["order_month"] = shipments["order_date"].dt.month

shipments["promised_days"] = (
    shipments["promised_delivery_date"] -
    shipments["order_date"]
).dt.days

categorical_features = [
    "carrier", "origin", "destination", "parcel_type"
]

numerical_features = [
    "order_day", "order_month", "promised_days", "weight_kg"
]

features = categorical_features + numerical_features

X = shipments[features]
y = shipments["is_late"]

# 6. Time-based split
shipments = shipments.sort_values("order_date").reset_index(drop=True)

X = shipments[features]
y = shipments["is_late"]

split_index = int(len(shipments) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("TRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# 7. Preprocessing
def create_preprocessor():
    return ColumnTransformer([
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features),

        ("numerical",
         StandardScaler(),
         numerical_features)
    ])

# 8. Models
models = {
    "Logistic Regression": Pipeline([
        ("preprocessing", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ("preprocessing", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])
}

# 9. Train and evaluate
results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    if name == "Logistic Regression":
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/parcel_delay_model.joblib")
        print("Logistic Regression model saved!")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n", name)
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["On Time", "Late"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["On Time", "Late"]
    )

    display.plot()
    plt.title(name + " Confusion Matrix")
    plt.savefig(
        "confusion_matrix_" + name.lower().replace(" ", "_") + ".png"
    )
    plt.close()

# 10. Model comparison
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

print("\nMODEL EVALUATION COMPLETED!")
