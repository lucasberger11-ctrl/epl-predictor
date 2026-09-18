import math
import streamlit as st

st.set_page_config(page_title="EPL Simple Predictor", page_icon="⚽", layout="centered")

TEAMS = [
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton & Hove Albion",
    "Burnley","Chelsea","Crystal Palace","Everton","Fulham","Leeds United",
    "Liverpool","Manchester City","Manchester United","Newcastle United",
    "Nottingham Forest","Sunderland","Tottenham Hotspur","West Ham United",
    "Wolverhampton Wanderers"
]

# Editable baseline expected goals. These are deliberately neutral defaults.
# Replace/update them with your own current data as you develop the model.
DEFAULT_HOME_XG = 1.55
DEFAULT_AWAY_XG = 1.25
DEFAULT_CORNERS = 10.0

def poisson_cdf(k, lam):
    return sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k + 1))

def over_half_line_probability(line, expected):
    # For X.5 markets: Over 2.5 = P(X >= 3)
    cutoff = math.floor(line)
    return 1.0 - poisson_cdf(cutoff, expected)

def pct(x):
    return f"{100*x:.1f}%"

st.title("⚽ EPL Simple Predictor")
st.caption("Simple probability calculator for goals and corners. No Kalshi prices are used.")

home = st.selectbox("Home team", TEAMS, index=0)
away_options = [t for t in TEAMS if t != home]
away = st.selectbox("Away team", away_options, index=0)

market = st.selectbox(
    "Prediction type",
    ["Total Match Goals", "Individual Team Goals", "Total Match Corners"]
)

st.divider()

if market == "Total Match Goals":
    line = st.selectbox("Goal line", [1.5, 2.5, 3.5, 4.5, 5.5])
    side = st.radio("Prediction", ["Over", "Under"], horizontal=True)
    home_xg = st.number_input("Expected home goals", 0.10, 5.00, DEFAULT_HOME_XG, 0.05)
    away_xg = st.number_input("Expected away goals", 0.10, 5.00, DEFAULT_AWAY_XG, 0.05)
    expected = home_xg + away_xg
    label = f"{side} {line} total goals"

elif market == "Individual Team Goals":
    team = st.selectbox("Team", [home, away])
    line = st.selectbox("Team goal line", [0.5, 1.5, 2.5, 3.5, 4.5])
    side = st.radio("Prediction", ["Over", "Under"], horizontal=True)
    default = DEFAULT_HOME_XG if team == home else DEFAULT_AWAY_XG
    expected = st.number_input(f"Expected goals — {team}", 0.10, 5.00, default, 0.05)
    label = f"{team}: {side} {line} goals"

else:
    line = st.selectbox("Corner line", [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
    side = st.radio("Prediction", ["Over", "Under"], horizontal=True)
    expected = st.number_input("Expected total corners", 1.0, 25.0, DEFAULT_CORNERS, 0.1)
    label = f"{side} {line} total corners"

if st.button("Calculate Probability", type="primary", use_container_width=True):
    p_over = over_half_line_probability(line, expected)
    p = p_over if side == "Over" else 1.0 - p_over
    opposite = 1.0 - p
    fair_decimal = 1.0 / p if p > 0 else float("inf")

    st.subheader(f"{home} vs {away}")
    st.write(f"**Prediction:** {label}")
    st.metric("Estimated probability", pct(p))
    st.write(f"Opposite side: **{pct(opposite)}**")
    st.write(f"Fair decimal odds: **{fair_decimal:.2f}**")
    st.caption(f"Model expectation used: {expected:.2f}")

st.divider()
st.caption(
    "Important: this starter version uses a Poisson probability model and user-editable "
    "expected goals/corners. The default expectations are placeholders, not live EPL estimates. "
    "The next upgrade can automatically calculate expectations from historical and recent team data."
)
