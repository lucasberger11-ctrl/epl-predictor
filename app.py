import math
import streamlit as st

st.set_page_config(page_title="EPL Probability Model", page_icon="⚽", layout="centered")

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(k + 1))

def over_half_line_probability(line, lam):
    # Designed for half-lines such as 2.5, 9.5.
    threshold = math.floor(line)
    return 1.0 - poisson_cdf(threshold, lam)

def blend(model_prob, market_prob, model_weight=0.70):
    if market_prob is None:
        return model_prob
    return model_weight * model_prob + (1-model_weight) * market_prob

def fair_prob_from_decimal_odds(odds):
    return 1 / odds if odds and odds > 1 else None

st.title("⚽ EPL Probability Model")
st.caption("Independent match model. Kalshi prices are for comparison only and are not used to generate the prediction.")

home = st.text_input("Home team", "Brighton")
away = st.text_input("Away team", "Arsenal")

st.subheader("Team estimates")
st.write("Enter your expected match rates from recent EPL/team data. These are the engine inputs.")
home_goals = st.number_input("Expected home goals", 0.0, 6.0, 1.25, 0.05)
away_goals = st.number_input("Expected away goals", 0.0, 6.0, 1.70, 0.05)
home_corners = st.number_input("Expected home corners", 0.0, 15.0, 4.6, 0.1)
away_corners = st.number_input("Expected away corners", 0.0, 15.0, 5.4, 0.1)

st.subheader("Your prediction")
market = st.selectbox("Market", ["Total Goals", "Total Corners", "Both Teams to Score", "First-Half Result"])

market_odds = st.number_input("Sportsbook decimal odds (optional; enter 0 to ignore)", 0.0, 20.0, 0.0, 0.01)
market_prob = fair_prob_from_decimal_odds(market_odds)
kalshi_price = st.number_input("Kalshi YES price in cents (comparison only; optional)", 0, 100, 0, 1)

raw = None
description = ""

if market == "Total Goals":
    line = st.number_input("Goals line (use .5)", 0.5, 8.5, 2.5, 1.0)
    side = st.selectbox("Side", ["Over", "Under"])
    over = over_half_line_probability(line, home_goals + away_goals)
    raw = over if side == "Over" else 1-over
    description = f"{side} {line} total goals"

elif market == "Total Corners":
    line = st.number_input("Corners line (use .5)", 0.5, 20.5, 9.5, 1.0)
    side = st.selectbox("Side", ["Over", "Under"])
    over = over_half_line_probability(line, home_corners + away_corners)
    raw = over if side == "Over" else 1-over
    description = f"{side} {line} total corners"

elif market == "Both Teams to Score":
    side = st.selectbox("BTTS", ["Yes", "No"])
    yes = (1-math.exp(-home_goals)) * (1-math.exp(-away_goals))
    raw = yes if side == "Yes" else 1-yes
    description = f"Both Teams to Score — {side}"

else:
    choice = st.selectbox("First-half result", ["Home", "Draw", "Away"])
    # Approximate first-half scoring as 45% of full-match expected goals.
    lh, la = home_goals * 0.45, away_goals * 0.45
    hp = dp = ap = 0.0
    for h in range(8):
        for a in range(8):
            p = poisson_pmf(h, lh) * poisson_pmf(a, la)
            if h > a: hp += p
            elif h == a: dp += p
            else: ap += p
    raw = {"Home": hp, "Draw": dp, "Away": ap}[choice]
    description = f"First-half result — {choice}"

if st.button("Calculate probability", type="primary"):
    final = blend(raw, market_prob, 0.70) if market_prob else raw
    st.metric("Statistical model probability", f"{raw*100:.1f}%")
    if market_prob:
        st.metric("70/30 blended expectation", f"{final*100:.1f}%")
        st.caption("30% market component uses the entered sportsbook odds. For a rigorous market input, use de-vigged odds.")
    else:
        st.metric("Expectation", f"{final*100:.1f}%")

    fair_decimal = 1/final if final > 0 else float("inf")
    st.write(f"**Prediction:** {home} vs {away} — {description}")
    st.write(f"**Model fair decimal odds:** {fair_decimal:.2f}")

    if kalshi_price > 0:
        kalshi_prob = kalshi_price / 100
        edge = final - kalshi_prob
        st.write(f"**Kalshi comparison:** {kalshi_price}¢ ≈ {kalshi_prob*100:.1f}% implied probability")
        st.write(f"**Model edge vs. Kalshi:** {edge*100:+.1f} percentage points")
        st.caption("A positive model edge is not a guarantee of profit; probabilities and input estimates can be wrong.")

st.divider()
st.caption("Educational probability tool — not financial advice. Improve accuracy by updating expected-goal and corner inputs from current EPL data.")
