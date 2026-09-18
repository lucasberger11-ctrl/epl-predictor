import math
import streamlit as st

st.set_page_config(page_title="EPL Match Predictor V5", page_icon="⚽", layout="centered")

# 2025-26 completed Premier League season baseline data.
# xG/xGA and corners per match are used so matchup selections produce team-specific probabilities.
TEAM_DATA = {
    "Arsenal": {"xg":66.13,"xga":29.98,"corners":5.68},
    "Aston Villa": {"xg":48.46,"xga":55.75,"corners":5.24},
    "Bournemouth": {"xg":62.93,"xga":55.67,"corners":5.58},
    "Brentford": {"xg":59.93,"xga":54.42,"corners":4.79},
    "Brighton & Hove Albion": {"xg":56.74,"xga":51.04,"corners":4.87},
    "Burnley": {"xg":33.43,"xga":74.99,"corners":3.66},
    "Chelsea": {"xg":66.74,"xga":52.49,"corners":5.95},
    "Crystal Palace": {"xg":57.64,"xga":54.96,"corners":4.18},
    "Everton": {"xg":47.45,"xga":56.91,"corners":4.42},
    "Fulham": {"xg":49.67,"xga":53.46,"corners":4.92},
    "Leeds United": {"xg":53.93,"xga":54.69,"corners":4.42},
    "Liverpool": {"xg":59.25,"xga":47.02,"corners":6.11},
    "Manchester City": {"xg":70.96,"xga":44.54,"corners":6.42},
    "Manchester United": {"xg":64.71,"xga":49.40,"corners":4.76},
    "Newcastle United": {"xg":58.12,"xga":51.28,"corners":6.05},
    "Nottingham Forest": {"xg":47.21,"xga":55.17,"corners":5.21},
    "Sunderland": {"xg":39.06,"xga":53.75,"corners":3.63},
    "Tottenham Hotspur": {"xg":42.53,"xga":50.90,"corners":5.50},
    "West Ham United": {"xg":45.92,"xga":60.91,"corners":4.92},
    "Wolverhampton Wanderers": {"xg":35.66,"xga":59.14,"corners":3.39},
}
TEAMS = list(TEAM_DATA.keys())
MATCHES = 38

# League baselines calculated from the 20 teams above.
LEAGUE_XG_PER_TEAM_MATCH = sum(v["xg"] for v in TEAM_DATA.values()) / (len(TEAM_DATA)*MATCHES)
LEAGUE_XGA_PER_TEAM_MATCH = sum(v["xga"] for v in TEAM_DATA.values()) / (len(TEAM_DATA)*MATCHES)

# Mild home/away multipliers. Team strengths remain the main driver.
HOME_FACTOR = 1.10
AWAY_FACTOR = 0.90

def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(k + 1))

def over_probability(line, expected):
    return 1.0 - poisson_cdf(math.floor(line), expected)

def pct(p):
    return f"{100*p:.1f}%"

def matchup_expected_goals(home, away):
    h = TEAM_DATA[home]
    a = TEAM_DATA[away]

    h_attack = (h["xg"]/MATCHES) / LEAGUE_XG_PER_TEAM_MATCH
    a_attack = (a["xg"]/MATCHES) / LEAGUE_XG_PER_TEAM_MATCH
    h_def_weakness = (h["xga"]/MATCHES) / LEAGUE_XGA_PER_TEAM_MATCH
    a_def_weakness = (a["xga"]/MATCHES) / LEAGUE_XGA_PER_TEAM_MATCH

    home_lambda = LEAGUE_XG_PER_TEAM_MATCH * h_attack * a_def_weakness * HOME_FACTOR
    away_lambda = LEAGUE_XG_PER_TEAM_MATCH * a_attack * h_def_weakness * AWAY_FACTOR

    # Keep extreme Poisson inputs in a sensible range.
    return max(0.25, min(home_lambda, 4.0)), max(0.20, min(away_lambda, 4.0))

def matchup_expected_corners(home, away):
    # Uses each club's actual 2025-26 corners-per-match production.
    # Small venue adjustment keeps total close to the two teams' combined historical rates.
    home_c = TEAM_DATA[home]["corners"] * 1.03
    away_c = TEAM_DATA[away]["corners"] * 0.97
    return home_c, away_c

def result_probabilities(home_lam, away_lam, max_goals=12):
    hw = dr = aw = 0.0
    for h in range(max_goals + 1):
        ph = poisson_pmf(h, home_lam)
        for a in range(max_goals + 1):
            p = ph * poisson_pmf(a, away_lam)
            if h > a: hw += p
            elif h == a: dr += p
            else: aw += p
    s = hw + dr + aw
    return hw/s, dr/s, aw/s

def ou_rows(lines, expected):
    rows = []
    for line in lines:
        ov = over_probability(line, expected)
        rows.append({"Line": f"{line:.1f}", "Over": pct(ov), "Under": pct(1-ov)})
    return rows

st.title("⚽ EPL Match Predictor V5")
st.caption("Same outcome format — now powered by team-specific 2025-26 EPL xG/xGA and corner rates.")

home = st.selectbox("Home team", TEAMS)
away = st.selectbox("Away team", [t for t in TEAMS if t != home])

hxg, axg = matchup_expected_goals(home, away)
total_goals = hxg + axg
home_c, away_c = matchup_expected_corners(home, away)
total_corners = home_c + away_c
home_win, draw, away_win = result_probabilities(hxg, axg)

st.header(f"{home} vs {away}")

st.subheader("🏆 Match result")
st.table([
    {"Outcome": f"{home} win", "Probability": pct(home_win)},
    {"Outcome": "Draw", "Probability": pct(draw)},
    {"Outcome": f"{away} win", "Probability": pct(away_win)},
])

st.subheader("📊 Model expectations")
a,b = st.columns(2)
a.metric(f"{home} predicted goals", f"{hxg:.2f}")
b.metric(f"{away} predicted goals", f"{axg:.2f}")
c,d = st.columns(2)
c.metric("Predicted total goals", f"{total_goals:.2f}")
d.metric("Predicted total corners", f"{total_corners:.2f}")

st.divider()
st.subheader("⚽ Total match goals — Over / Under")
st.table(ou_rows([0.5,1.5,2.5,3.5,4.5,5.5,6.5], total_goals))

st.subheader(f"🥅 {home} goals — Over / Under")
st.table(ou_rows([0.5,1.5,2.5,3.5,4.5,5.5], hxg))

st.subheader(f"🥅 {away} goals — Over / Under")
st.table(ou_rows([0.5,1.5,2.5,3.5,4.5,5.5], axg))

st.subheader("🚩 Total corners — Over / Under")
st.table(ou_rows(
    [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5],
    total_corners
))

st.divider()
st.caption(
    "Data baseline: completed 2025-26 Premier League season. Goal probabilities use team xG/xGA "
    "strengths with a Poisson model; corner probabilities use team corners-per-match rates with a "
    "Poisson model. These are model estimates, not guarantees or betting-market prices."
)
