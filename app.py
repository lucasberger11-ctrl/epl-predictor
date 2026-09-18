import math
import io
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

st.set_page_config(page_title="EPL ML Predictor", page_icon="⚽", layout="centered")

SEASONS = ["2021-22","2022-23","2023-24","2024-25","2025-26","2026-27"]
SEASON_CODES = {"2021-22":"2122","2022-23":"2223","2023-24":"2324",
                "2024-25":"2425","2025-26":"2526","2026-27":"2627"}
BASE = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
ROLL = 8

@st.cache_data(ttl=3600)
def load_data():
    frames = []
    for season in SEASONS:
        try:
            r = requests.get(BASE.format(SEASON_CODES[season]), timeout=15)
            r.raise_for_status()
            d = pd.read_csv(io.BytesIO(r.content))
            d["Season"] = season
            frames.append(d)
        except Exception:
            pass
    if not frames:
        raise RuntimeError("Could not download EPL training data.")
    df = pd.concat(frames, ignore_index=True)
    needed = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HC","AC","HS","AS","HST","AST"]
    for c in needed:
        if c not in df: df[c] = np.nan
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]).sort_values("Date").reset_index(drop=True)
    return df

def pre_match_features(df):
    histories = {}
    rows = []
    defaults = dict(gf=1.35, ga=1.35, corners=5.0, corners_against=5.0,
                    shots=12.0, sot=4.0, points=1.35)
    def avg(team, key):
        vals = histories.get(team, {}).get(key, [])
        return float(np.mean(vals[-ROLL:])) if vals else defaults[key]
    for _, r in df.iterrows():
        h,a = r.HomeTeam,r.AwayTeam
        row = {
            "Date":r.Date,"Season":r.Season,"HomeTeam":h,"AwayTeam":a,
            "h_gf":avg(h,"gf"),"h_ga":avg(h,"ga"),"h_c":avg(h,"corners"),
            "h_ca":avg(h,"corners_against"),"h_sh":avg(h,"shots"),
            "h_sot":avg(h,"sot"),"h_pts":avg(h,"points"),
            "a_gf":avg(a,"gf"),"a_ga":avg(a,"ga"),"a_c":avg(a,"corners"),
            "a_ca":avg(a,"corners_against"),"a_sh":avg(a,"shots"),
            "a_sot":avg(a,"sot"),"a_pts":avg(a,"points"),
            "FTR":r.FTR,"FTHG":float(r.FTHG),"FTAG":float(r.FTAG),
            "TotalCorners":float(r.HC+r.AC) if pd.notna(r.HC) and pd.notna(r.AC) else np.nan
        }
        rows.append(row)
        hp = 3 if r.FTR=="H" else 1 if r.FTR=="D" else 0
        ap = 3 if r.FTR=="A" else 1 if r.FTR=="D" else 0
        vals = {
            h: dict(gf=r.FTHG,ga=r.FTAG,corners=r.HC,corners_against=r.AC,shots=r.HS,sot=r.HST,points=hp),
            a: dict(gf=r.FTAG,ga=r.FTHG,corners=r.AC,corners_against=r.HC,shots=r.AS,sot=r.AST,points=ap)
        }
        for team, ds in vals.items():
            histories.setdefault(team,{k:[] for k in defaults})
            for k,v in ds.items():
                if pd.notna(v): histories[team][k].append(float(v))
    return pd.DataFrame(rows), histories

FEATURES = ["h_gf","h_ga","h_c","h_ca","h_sh","h_sot","h_pts",
            "a_gf","a_ga","a_c","a_ca","a_sh","a_sot","a_pts"]

@st.cache_resource(ttl=3600)
def train_models():
    raw = load_data()
    feat, histories = pre_match_features(raw)
    clean = feat.dropna(subset=FEATURES+["FTR","FTHG","FTAG","TotalCorners"]).copy()
    # Time-based split: latest 20% is held out; no random future leakage.
    cut = int(len(clean)*0.80)
    train, test = clean.iloc[:cut], clean.iloc[cut:]
    Xtr, Xte = train[FEATURES], test[FEATURES]

    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(Xtr, train.FTR)
    hg = HistGradientBoostingRegressor(max_iter=250, l2_regularization=1.0, random_state=42)
    ag = HistGradientBoostingRegressor(max_iter=250, l2_regularization=1.0, random_state=43)
    cr = HistGradientBoostingRegressor(max_iter=250, l2_regularization=1.0, random_state=44)
    hg.fit(Xtr, train.FTHG); ag.fit(Xtr, train.FTAG); cr.fit(Xtr, train.TotalCorners)

    probs = clf.predict_proba(Xte)
    metrics = {
        "accuracy": accuracy_score(test.FTR, clf.predict(Xte)),
        "logloss": log_loss(test.FTR, probs, labels=clf.classes_),
        "goal_mae": (mean_absolute_error(test.FTHG,hg.predict(Xte))+
                     mean_absolute_error(test.FTAG,ag.predict(Xte)))/2,
        "corner_mae": mean_absolute_error(test.TotalCorners,cr.predict(Xte)),
        "train_n":len(train),"test_n":len(test)
    }
    return raw, histories, clf, hg, ag, cr, metrics

def current_features(team, histories, key, default):
    vals=histories.get(team,{}).get(key,[])
    return float(np.mean(vals[-ROLL:])) if vals else default

def make_match_row(home,away,histories):
    d={"gf":1.35,"ga":1.35,"corners":5.0,"corners_against":5.0,"shots":12.0,"sot":4.0,"points":1.35}
    return pd.DataFrame([{
        "h_gf":current_features(home,histories,"gf",d["gf"]),
        "h_ga":current_features(home,histories,"ga",d["ga"]),
        "h_c":current_features(home,histories,"corners",d["corners"]),
        "h_ca":current_features(home,histories,"corners_against",d["corners_against"]),
        "h_sh":current_features(home,histories,"shots",d["shots"]),
        "h_sot":current_features(home,histories,"sot",d["sot"]),
        "h_pts":current_features(home,histories,"points",d["points"]),
        "a_gf":current_features(away,histories,"gf",d["gf"]),
        "a_ga":current_features(away,histories,"ga",d["ga"]),
        "a_c":current_features(away,histories,"corners",d["corners"]),
        "a_ca":current_features(away,histories,"corners_against",d["corners_against"]),
        "a_sh":current_features(away,histories,"shots",d["shots"]),
        "a_sot":current_features(away,histories,"sot",d["sot"]),
        "a_pts":current_features(away,histories,"points",d["points"]),
    }])[FEATURES]

def poisson_cdf(k,lam):
    lam=max(float(lam),0.05)
    return sum(math.exp(-lam)*lam**i/math.factorial(i) for i in range(k+1))
def ou_prob(line,mean):
    return max(0,min(1,1-poisson_cdf(math.floor(line),mean)))
def pct(x): return f"{100*x:.1f}%"
def ou_table(lines, mean):
    return [{"Line":f"{x:.1f}","Over":pct(ou_prob(x,mean)),"Under":pct(1-ou_prob(x,mean))} for x in lines]

st.title("⚽ EPL Machine-Learning Predictor")
st.caption("Trains on historical EPL matches and recent team form. Match-result probabilities come from a Random Forest; goals and corners use gradient-boosted regression.")

try:
    raw,histories,clf,hg,ag,cr,metrics=train_models()
except Exception as e:
    st.error(f"Training data could not be loaded: {e}")
    st.stop()

teams=sorted(set(raw.HomeTeam).union(raw.AwayTeam))
home=st.selectbox("Home team",teams,index=teams.index("Arsenal") if "Arsenal" in teams else 0)
away_opts=[x for x in teams if x!=home]
away=st.selectbox("Away team",away_opts,index=0)

X=make_match_row(home,away,histories)
p=clf.predict_proba(X)[0]
prob=dict(zip(clf.classes_,p))
home_g=max(0.05,float(hg.predict(X)[0]))
away_g=max(0.05,float(ag.predict(X)[0]))
corners=max(0.1,float(cr.predict(X)[0]))
total_g=home_g+away_g

st.header(f"{home} vs {away}")
st.subheader("🏆 Match result — ML probabilities")
st.table([
    {"Outcome":f"{home} win","Probability":pct(prob.get("H",0))},
    {"Outcome":"Draw","Probability":pct(prob.get("D",0))},
    {"Outcome":f"{away} win","Probability":pct(prob.get("A",0))}
])

st.subheader("📊 ML expectations")
a,b=st.columns(2); a.metric(f"{home} goals",f"{home_g:.2f}"); b.metric(f"{away} goals",f"{away_g:.2f}")
c,d=st.columns(2); c.metric("Total goals",f"{total_g:.2f}"); d.metric("Total corners",f"{corners:.2f}")

st.subheader("⚽ Total goals — Over / Under")
st.table(ou_table([0.5,1.5,2.5,3.5,4.5,5.5],total_g))
st.subheader(f"🥅 {home} goals — Over / Under")
st.table(ou_table([0.5,1.5,2.5,3.5,4.5],home_g))
st.subheader(f"🥅 {away} goals — Over / Under")
st.table(ou_table([0.5,1.5,2.5,3.5,4.5],away_g))
st.subheader("🚩 Total corners — Over / Under")
st.table(ou_table([5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5],corners))

with st.expander("🧪 Model test results"):
    st.write(f"Training matches: **{metrics['train_n']}** | Held-out matches: **{metrics['test_n']}**")
    st.write(f"Result accuracy: **{metrics['accuracy']:.1%}**")
    st.write(f"Result log loss: **{metrics['logloss']:.3f}**")
    st.write(f"Goals MAE (per team): **{metrics['goal_mae']:.2f}**")
    st.write(f"Total corners MAE: **{metrics['corner_mae']:.2f}**")
    st.caption("The held-out test is chronological: the model trains on earlier games and is evaluated on later unseen games.")

st.divider()
st.caption("Probabilities are model estimates, not guarantees. Over/under probabilities use a Poisson distribution centered on the ML-predicted mean.")
