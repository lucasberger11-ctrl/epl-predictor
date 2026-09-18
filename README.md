# EPL Probability Model

A simple Streamlit app for estimating probabilities for:
- Total goals (Over/Under half-lines)
- Total corners (Over/Under half-lines)
- Both Teams to Score
- First-half result

The statistical model is independent of Kalshi. An optional sportsbook decimal-odds input can be blended 70% model / 30% market. Kalshi's YES price is only compared after the prediction is calculated.

## Laptop setup

1. Install Python 3.11+.
2. Unzip this folder.
3. Open Terminal / Command Prompt in the folder.
4. Install requirements:

   pip install -r requirements.txt

5. Start the app:

   streamlit run app.py

Your browser should open the program automatically.

## iPhone

The easiest reliable setup is to run the Streamlit app from a computer/cloud Python host and open its web address in Safari. iOS does not run a normal Streamlit/Python desktop app directly from the Files app.

## Inputs

The included version lets you enter expected home/away goals and corners. These should be updated from current EPL data before using a match prediction.

Sportsbook odds are optional. For serious use, convert both sides of a sportsbook market to de-vigged probabilities rather than entering a single raw implied probability.

## Important

This is a probability estimator, not a guarantee of outcomes or profit.
