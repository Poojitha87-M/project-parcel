
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

carriers = ["FastTrack", "QuickShip", "SpeedGo"]
origins = ["Hyderabad", "Bangalore", "Chennai", "Guntur", "Vijayawada"]
destinations = ["Hyderabad", "Bangalore", "Chennai", "Guntur", "Vijayawada"]
parcel_types = ["Standard", "Express", "Heavy"]

records = []

start_date = datetime(2025, 1, 1)

for i in range(500):
    order_date = start_date + timedelta(days=i)

    carrier = random.choice(carriers)
    origin = random.choice(origins)
    destination = random.choice(destinations)
    while destination == origin:
        destination = random.choice(destinations)

    parcel_type = random.choice(parcel_types)
    weight = round(random.uniform(0.5, 20), 2)

    promised_days = random.randint(2, 5)
    promised_date = order_date + timedelta(days=promised_days)

    # Simulate realistic delivery-delay risk
    risk = 0.20

    if carrier == "QuickShip":
        risk += 0.12
    elif carrier == "SpeedGo":
        risk -= 0.05

    if parcel_type == "Heavy":
        risk += 0.15
    elif parcel_type == "Express":
        risk -= 0.10

    if weight > 10:
        risk += 0.10

    if origin in ["Guntur", "Vijayawada"]:
        risk += 0.05

    risk = max(0.05, min(risk, 0.85))

    if random.random() < risk:
        delay_days = random.randint(1, 4)
    else:
        delay_days = random.choice([-1, 0])

    actual_date = promised_date + timedelta(days=delay_days)

    records.append({
        "shipment_id": f"S{i+1:04d}",
        "order_date": order_date.strftime("%Y-%m-%d"),
        "promised_delivery_date": promised_date.strftime("%Y-%m-%d"),
        "actual_delivery_date": actual_date.strftime("%Y-%m-%d"),
        "carrier": carrier,
        "origin": origin,
        "destination": destination,
        "parcel_type": parcel_type,
        "weight_kg": weight
    })

df = pd.DataFrame(records)

df.to_csv("data/shipments.csv", index=False)

print("Synthetic shipment dataset generated!")
print("Total records:", len(df))
print("\nLate and on-time distribution:")

df["is_late"] = (
    pd.to_datetime(df["actual_delivery_date"]) >
    pd.to_datetime(df["promised_delivery_date"])
).astype(int)

print(df["is_late"].value_counts())
print("\nDataset preview:")
print(df.head())
