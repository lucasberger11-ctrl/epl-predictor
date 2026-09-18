# EPL Simple Predictor

A deliberately simple Streamlit application for three strict EPL markets:

- Total Match Goals: Over/Under 1.5, 2.5, 3.5, 4.5, 5.5
- Individual Team Goals: Over/Under 0.5, 1.5, 2.5, 3.5, 4.5
- Total Match Corners: Over/Under 5.5 through 14.5

## How it works
The app converts an expected number of goals/corners into probabilities with a Poisson model.
Kalshi prices are NOT used.

The initial expected-value fields are editable placeholders. This avoids pretending that static
numbers are current team predictions. A later version can automatically generate those expected
values from EPL historical/recent data.

## Run locally
1. Install Python.
2. In this folder run:
   pip install -r requirements.txt
3. Then:
   streamlit run app.py

## Streamlit Community Cloud
Upload `app.py`, `requirements.txt`, and this README to the ROOT of your GitHub repository.
Set the main file path to:
app.py
