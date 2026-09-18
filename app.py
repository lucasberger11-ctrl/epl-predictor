import math
import streamlit as st

st.set_page_config(page_title="EPL Match Predictor", page_icon="⚽", layout="centered")

TEAMS = [
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton & Hove Albion",
    "Burnley","Chelsea","Crystal Palace","Everton","Fulham","Leeds United",
    "Liverpool","Manchester City","Manchester United","Newcastle United",
    "Nottingham Forest","Sunderland","Tottenham Hotspur","West Ham United",
    "Wolverhampton Wanderers"
]

# Neutral starter inputs. Replace with a current EPL data feed/model in the next version.
ATTACK = {t: 1.0 for t in TEAMS}
DEFENSE = {t: 1.0 for t in TEAMS}
CORNERS_FOR = {t: 5.0 for t in TEAMS}
CORNERS_AGAINST = {t: 5.0 for t in TEAMS}
LEAGUE_HOME_GOALS = 1.55
LEAGUE_AWAY_GOALS = 1.25

def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(k + 1))

def over_prob(line, expected):
    return 1 - poisson_cdf(math.floor(line), expected)

def pct(p):
    return f"{100*p:.1f}%"

def expected_goals(home, away):
    hxg = LEAGUE_HOME_GOALS * ATTACK[home] * DEFENSE[away]
    axg = LEAGUE_AWAY_GOALS * ATTACK[away] * DEFENSE[home]
    return hxg, axg

def expected_corners(home, away):
    hc = (CORNERS_FOR[home] + CORNERS_AGAINST[away]) / 2
    ac = (CORNERS_FOR[away] + CORNERS_AGAINST[home]) / 2
    return hc, ac

def ou_rows(lines, expected):
    out = []
    for line in lines:
        over = over_prob(line, expected)
        out.append({"Line": f"{line:.1f}", "Over": pct(over), "Under": pct(1-over)})
    return out

def result_probabilities(home_lambda, away_lambda, max_goals=12):
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        ph = poisson_pmf(h, home_lambda)
        for a in range(max_goals + 1):
            p = ph * poisson_pmf(a, away_lambda)
            if h > a: home_win += p
            elif h == a: draw += p
            else: away_win += p
    total = home_win + draw + away_win
    return home_win/total, draw/total, away_win/total

st.title("⚽ EPL Match Predictor")
st.caption("One matchup → all core probabilities. No Kalshi market percentage is used.")

home = st.selectbox("Home team", TEAMS)
away = st.selectbox("Away team", [t for t in TEAMS if t != home])

hxg, axg = expected_goals(home, away)
txg = hxg + axg
hc, ac = expected_corners(home, away)
tc = hc + ac
hp, dp, ap = result_probabilities(hxg, axg)

st.header(f"{home} vs {away}")

st.subheader("🏆 Match result")
result_rows = [
    {"Outcome": f"{home} win", "Probability": pct(hp)},
    {"Outcome": "Draw", "Probability": pct(dp)},
    {"Outcome": f"{away} win", "Probability": pct(ap)},
]
st.table(result_rows)

most_likely = max(
    [(f"{home} win", hp), ("Draw", dp), (f"{away} win", ap)],
    key=lambda x: x[1]
)
st.metric("Highest model probability", most_likely[0], pct(most_likely[1]))

st.subheader("📊 Model expectations")
a,b = st.columns(2)
a.metric(f"{home} predicted goals", f"{hxg:.2f}")
b.metric(f"{away} predicted goals", f"{axg:.2f}")
c,d = st.columns(2)
c.metric("Predicted total goals", f"{txg:.2f}")
d.metric("Predicted total corners", f"{tc:.2f}")

st.divider()
st.subheader("⚽ Total match goals — Over / Under")
st.table(ou_rows([0.5,1.5,2.5,3.5,4.5,5.5,6.5], txg))

st.subheader(f"🥅 {home} goals — Over / Under")
st.table(ou_rows([0.5,1.5,2.5,3.5,4.5,5.5], hxg))

st.subheader(f"🥅 {away} goals — Over / Under")
st.table(ou_rows([0.5,1.5,2.5,3.5,4.5,5.5], axg))

st.subheader("🚩 Total corners — Over / Under")
st.table(ou_rows([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5], tc))

st.divider()
st.caption(
    "V4 interface/probability engine. Team-strength and corner inputs are still neutral placeholders, "
    "not live EPL estimates. The next data-connected version should derive them from real historical "
    "and recent team performance before these percentages are used for decision-making."
)
