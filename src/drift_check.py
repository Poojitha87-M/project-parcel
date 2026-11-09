
import pandas as pd

training = pd.read_csv("data/shipments.csv")

# Demonstration: use a later portion as incoming data
incoming = training.tail(100)

training_mean = training["weight_kg"].mean()
incoming_mean = incoming["weight_kg"].mean()

difference = abs(training_mean - incoming_mean)

print("TRAINING AVERAGE WEIGHT:", round(training_mean, 2))
print("INCOMING AVERAGE WEIGHT:", round(incoming_mean, 2))

if difference > 2:
    print("WARNING: Possible data drift detected!")
else:
    print("No significant weight drift detected.")
