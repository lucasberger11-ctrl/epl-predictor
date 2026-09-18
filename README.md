# EPL ML Predictor

This version is a genuine machine-learning application.

- Downloads historical English Premier League match data from Football-Data.co.uk.
- Builds pre-match rolling features from each team's previous 8 matches.
- RandomForestClassifier predicts Home / Draw / Away probabilities.
- HistGradientBoostingRegressor predicts home goals, away goals, and total corners.
- Uses a chronological 80/20 train/test split to reduce future-data leakage.
- Streamlit displays the same core outcomes as the earlier EPL predictor.

## Deploy
Use `app.py` as the Streamlit main file and `requirements.txt` as the dependency file.

The first app load trains the models and caches them. Data are refreshed periodically through Streamlit caching.
