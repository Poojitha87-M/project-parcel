# Day 3 Check-in – Project Parcel

## Objective
Package the prediction workflow and check basic reliability.

## Work Completed
- Saved the trained Logistic Regression model.
- Created a command-line prediction script.
- Added validation for carrier, location, parcel type, dates and weight.
- Added a basic data drift check using average parcel weight.
- Created automated tests for model loading, prediction, probability and input columns.
- Created README.md and FAILURES.md.

## Key Learning
A machine learning project needs more than model training. It also needs input validation, testing, monitoring and clear documentation.

## Limitations
The drift check is basic, and the project is a local demonstration using synthetic data. Further work is required before production use.