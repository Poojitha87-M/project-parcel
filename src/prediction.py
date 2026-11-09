
import joblib
import pandas as pd

model = joblib.load("models/parcel_delay_model.joblib")

print("PARCEL DELIVERY PREDICTION")
print("--------------------------")

# Valid values
valid_carriers = ["FastTrack", "QuickShip", "SpeedGo"]
valid_locations = [
    "Hyderabad", "Bangalore", "Chennai",
    "Guntur", "Vijayawada"
]
valid_types = ["Standard", "Express", "Heavy"]

try:
    carrier = input("Enter carrier: ").strip()
    origin = input("Enter origin: ").strip()
    destination = input("Enter destination: ").strip()
    parcel_type = input("Enter parcel type: ").strip()

    order_day = int(input("Enter order day (1-31): "))
    order_month = int(input("Enter order month (1-12): "))
    promised_days = int(input("Enter promised delivery days: "))
    weight_kg = float(input("Enter parcel weight in kg: "))

    # Validate values
    if carrier not in valid_carriers:
        raise ValueError("Invalid carrier name.")

    if origin not in valid_locations or destination not in valid_locations:
        raise ValueError("Invalid origin or destination.")

    if origin == destination:
        raise ValueError("Origin and destination cannot be the same.")

    if parcel_type not in valid_types:
        raise ValueError("Invalid parcel type.")

    if not 1 <= order_day <= 31:
        raise ValueError("Order day must be between 1 and 31.")

    if not 1 <= order_month <= 12:
        raise ValueError("Order month must be between 1 and 12.")

    if promised_days <= 0:
        raise ValueError("Promised days must be positive.")

    if weight_kg <= 0:
        raise ValueError("Weight must be positive.")

    # Prepare input
    new_shipment = pd.DataFrame([{
        "carrier": carrier,
        "origin": origin,
        "destination": destination,
        "parcel_type": parcel_type,
        "order_day": order_day,
        "order_month": order_month,
        "promised_days": promised_days,
        "weight_kg": weight_kg
    }])

    # Predict
    prediction = model.predict(new_shipment)[0]
    probability = model.predict_proba(new_shipment)[0][1]

    print("\nPREDICTION RESULT")

    if prediction == 1:
        print("Warning: Parcel may be LATE.")
    else:
        print("Parcel is predicted to be ON TIME.")

    print("Late delivery probability:", round(probability * 100, 2), "%")

except ValueError as error:
    print("\nINPUT ERROR:", error)

except Exception as error:
    print("\nPREDICTION ERROR:", error)
