# Project Parcel – Shipment Delay Prediction

## 1. Project Overview
Project Parcel is a machine learning demonstration that predicts whether a parcel shipment may be delivered late or on time.

The project uses Python, Pandas and Scikit-learn to prepare shipment data, train classification models, evaluate predictions and monitor a basic data drift indicator.

## 2. Dataset
The project uses 500 synthetically generated shipment records for demonstration. The records are not real company shipment data.

Features include:
- Carrier
- Origin and destination
- Parcel type
- Parcel weight
- Order day and month
- Promised delivery days

## 3. Target Definition
A shipment is labelled late (1) when its actual delivery date is later than its promised delivery date. Otherwise, it is labelled on time (0).

Actual delivery date is used only to create the target label. It is not used as a prediction feature.

## 4. Machine Learning Models
Two classification models were compared:
- Logistic Regression
- Random Forest Classifier

A chronological 80:20 train-test split was used to simulate predicting on later shipments.

## 5. Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Logistic Regression performed better than Random Forest on the test set. However, its late-shipment recall was approximately 24.2%, so many late shipments were missed.

## 6. Prediction Tool
A command-line prediction script accepts shipment details, validates inputs and displays:
- Predicted delivery status
- Late-delivery probability

## 7. Data Drift Check
A basic drift script compares average parcel weight in the training data with a later sample. This is a demonstration and does not provide complete production drift monitoring.

## 8. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the demonstration dataset:

```bash
python src/generate_data.py
```

Train and evaluate models:

```bash
python src/data_loading.py
```

Run a parcel prediction:

```bash
python src/prediction.py
```

Run the drift check:

```bash
python src/drift_check.py
```

Run automated tests:

```bash
python -m pytest tests/ -v
```

## 9. Project Structure

```text
Project-Parcel/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── generate_data.py
│   ├── data_loading.py
│   ├── prediction.py
│   └── drift_check.py
├── tests/
│   └── test_prediction.py
├── FAILURES.md
├── README.md
└── requirements.txt
```

## 10. Limitations and Future Improvements
The dataset is synthetic, late-shipment recall is low, and drift monitoring is limited to average parcel weight. Future work includes using real shipment data, improving late detection, monitoring more features and deploying the model as a service.