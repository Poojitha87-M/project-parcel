# Project Parcel – Shipment Delay Prediction

**Project:** Machine Learning for Parcel Delivery Prediction  
**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib  
**Models:** Logistic Regression and Random Forest

## 1. Problem Statement

Late parcel deliveries can affect customer satisfaction and logistics operations. The objective of this project is to build a machine learning demonstration that predicts whether a parcel may arrive late or on time using shipment information available when an order is placed.

A shipment is labelled:
- **Late (1):** Actual delivery date is after the promised delivery date.
- **On Time (0):** Actual delivery date is on or before the promised delivery date.

## 2. Dataset and Preparation

The project uses 500 synthetically generated shipment records. These records are created for demonstration and are not real company shipment data.

The dataset includes carrier, origin, destination, parcel type, weight, order date and promised delivery information.

Data preparation included:
- Converting date columns into date formats.
- Removing records with missing required dates.
- Removing duplicate shipment IDs.
- Excluding non-positive parcel weights.
- Creating the late-delivery target.

Actual delivery date was used only to create the target label. It was excluded from the model input features to avoid target leakage.

## 3. Feature Engineering and Model Development

The following features were used:
- Carrier
- Origin and destination
- Parcel type
- Order day and month
- Promised delivery days
- Parcel weight

Categorical features were transformed using One-Hot Encoding. Numerical features were scaled using StandardScaler.

The dataset was sorted by order date and divided chronologically:
- Training data: 400 records (80%)
- Testing data: 100 records (20%)

Two classification models were trained:
1. Logistic Regression
2. Random Forest Classifier

## 4. Model Evaluation

The models were evaluated using accuracy, precision, recall, F1-score and confusion matrices.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.69 | 0.64 |
| Late precision | 0.571 | 0.000 |
| Late recall | 0.242 | 0.000 |
| Late F1-score | 0.340 | 0.000 |

Logistic Regression performed better overall on this test set. However, its late-shipment recall was only 24.2%. Random Forest did not identify any late shipments in the test set.

These results show that accuracy alone is not sufficient when the business objective is to identify delayed shipments. Missing late shipments is an important limitation.

## 5. Prediction and Reliability

A command-line prediction script was developed using the saved Logistic Regression model.

It:
- Accepts shipment details from the user.
- Validates carrier, location, parcel type, day, month, promised days and weight.
- Predicts whether a shipment may be late or on time.
- Displays the estimated late-delivery probability.

A basic drift check compares average parcel weight in the training dataset with a later sample. It is a demonstration only and does not monitor all features or model performance.

Four automated tests were created to check model loading, prediction output, probability range and input columns.

## 6. Limitations and Risks

- The dataset is synthetic and does not represent real shipment operations.
- Late-shipment recall is low.
- The default prediction threshold is 0.5.
- The drift check monitors only average parcel weight.
- Individual prediction explanations are not implemented.
- The project is a local demonstration, not a production deployment.

## 7. Future Improvements

- Train and evaluate using real shipment records.
- Improve late-shipment recall and validate prediction thresholds.
- Analyse errors by carrier, region and parcel type.
- Monitor multiple features and model performance over time.
- Add explanations for individual predictions.
- Deploy the model as an API and introduce automated monitoring.

## 8. Conclusion

This project demonstrates an end-to-end basic machine learning workflow for parcel delay prediction, including data preparation, feature engineering, model comparison, prediction, input validation, drift checking, automated tests and documentation.

The results are preliminary because the dataset is synthetic and late-shipment detection remains weak. Real shipment data and further validation are necessary before operational use.