# Known Failures and Limitations

## 1. Synthetic Dataset
The project uses 500 synthetically generated shipment records, not real company data.

## 2. Low Late-Delivery Recall
Logistic Regression achieved 24.2% recall for late shipments. Many late shipments were missed.

## 3. Random Forest Limitation
Random Forest did not identify any late shipments in the test set.

## 4. Basic Drift Monitoring
The drift check only compares average parcel weight. It does not monitor all features.

## 5. Production Limitations
This is a local demonstration. It does not include a live API or automatic model retraining.

## Future Improvements
- Use real shipment data.
- Improve late-delivery detection.
- Monitor more features.
- Add explanations for predictions.
- Deploy the model as a service.