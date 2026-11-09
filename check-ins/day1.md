# Day 1 Check-in – Project Parcel

## Objective
Understand the shipment delay prediction problem and prepare the dataset.

## Work Completed
- Defined a late shipment as one delivered after its promised delivery date.
- Inspected shipment data for missing values, duplicate records and invalid weights.
- Checked carrier and warehouse reference information.
- Selected features available at prediction time.
- Created the `is_late` target column using actual and promised delivery dates.

## Key Learning
Actual delivery date is used to create the target label, not as a model input, to avoid data leakage.

## Risks Identified
- Limited dataset size.
- Missing or invalid shipment information could affect predictions.
- Historical patterns may not represent future shipment conditions.