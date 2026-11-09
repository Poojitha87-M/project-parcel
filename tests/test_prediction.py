
import joblib
import pandas as pd

model = joblib.load("models/parcel_delay_model.joblib")

def sample_shipment():
    return pd.DataFrame([{
        "carrier": "QuickShip",
        "origin": "Hyderabad",
        "destination": "Chennai",
        "parcel_type": "Heavy",
        "order_day": 10,
        "order_month": 5,
        "promised_days": 3,
        "weight_kg": 15.0
    }])

def test_model_loads():
    assert model is not None

def test_prediction_is_valid():
    prediction = model.predict(sample_shipment())
    assert prediction[0] in [0, 1]

def test_probability_is_valid():
    probability = model.predict_proba(sample_shipment())[0][1]
    assert 0 <= probability <= 1

def test_prediction_input_columns():
    expected_columns = [
        "carrier", "origin", "destination", "parcel_type",
        "order_day", "order_month", "promised_days", "weight_kg"
    ]
    assert list(sample_shipment().columns) == expected_columns
