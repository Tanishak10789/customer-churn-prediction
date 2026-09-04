"""
Streamlit UI for the Customer Churn Prediction pipeline.

Run from the project root:
    streamlit run app.py

Because train.py saved the ENTIRE pipeline (feature preprocessing + model) as one
joblib file, this app only has to collect raw customer fields and call
predict_proba. There is no separate scaler and no manual one-hot encoding here --
the pipeline handles all of that internally, exactly as it did during training.
"""

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH
from src.features import add_features

# --------------------------------------------------------------------------
# page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Console",
    page_icon="◆",
    layout="wide",
)

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@700;800&display=swap');

  :root{
    --bg:#080B14; --panel:#101829; --panel2:#0C1220;
    --line:#1E2A45; --line2:#28375A;
    --text:#E6ECF7; --mid:#9AABC9; --muted:#6E80A3;
    --safe:#35E0C8; --warn:#FFB13D; --risk:#FF4D6D; --violet:#7C5CFF;
  }

  html, body, .stApp, [class*="css"]{ font-family:'Inter',-apple-system,sans-serif; }
  .stApp{
    background:
      radial-gradient(900px 500px at 12% -8%, rgba(124,92,255,.16), transparent 60%),
      radial-gradient(800px 450px at 92% 0%, rgba(53,224,200,.10), transparent 60%),
      var(--bg);
  }
  #MainMenu, footer, header[data-testid="stHeader"]{display:none;}
  .block-container{padding-top:2rem; padding-bottom:3.5rem; max-width:1200px;}

  /* ---------------- hero ---------------- */
  .hero{
    position:relative; overflow:hidden; border-radius:18px; padding:26px 30px;
    background:linear-gradient(135deg, rgba(124,92,255,.14), rgba(53,224,200,.06) 55%, transparent);
    border:1px solid var(--line2); margin-bottom:24px;
  }
  .hero:after{
    content:''; position:absolute; inset:0;
    background-image:linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px),
                     linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
    background-size:34px 34px; pointer-events:none;
  }
  .hero .eyebrow{
    font-family:'JetBrains Mono',monospace; font-size:10.5px; font-weight:700;
    letter-spacing:.24em; text-transform:uppercase; color:var(--safe);
  }
  .hero h1{
    font-family:'Space Grotesk',sans-serif; font-size:40px; font-weight:700;
    letter-spacing:-.03em; margin:8px 0 8px; line-height:1.02; color:#fff;
  }
  .hero h1 em{
    font-style:normal;
    background:linear-gradient(92deg,var(--safe),var(--violet));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }
  .hero p{font-size:14.5px; color:var(--mid); margin:0; max-width:64ch; line-height:1.6;}

  /* ---------------- section labels ---------------- */
  .sectionlabel{
    font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:700;
    letter-spacing:.2em; text-transform:uppercase; color:var(--muted);
    padding-bottom:9px; margin-bottom:15px;
    border-bottom:1px solid var(--line);
  }
  .sectionlabel:before{content:'// '; color:var(--safe);}

  /* ---------------- widgets ---------------- */
  .stSelectbox label, .stSlider label, .stNumberInput label,
  div[data-testid="stWidgetLabel"] label, div[data-testid="stWidgetLabel"] p{
    color:var(--mid) !important; font-size:12px !important; font-weight:600 !important;
    letter-spacing:.02em !important; margin-bottom:3px !important;
  }
  div[data-baseweb="select"] > div{
    background:var(--panel2) !important; border:1px solid var(--line) !important;
    border-radius:10px !important; font-size:14px !important; color:var(--text) !important;
    min-height:44px !important;
  }
  div[data-baseweb="select"] > div:hover{border-color:var(--safe) !important;}
  div[data-baseweb="popover"] div{background:var(--panel) !important; color:var(--text) !important;}
  .stNumberInput input{
    background:var(--panel2) !important; color:var(--text) !important;
    font-family:'JetBrains Mono',monospace !important; font-size:14px !important;
  }
  .stNumberInput div[data-baseweb="input"]{
    border:1px solid var(--line) !important; border-radius:10px !important;
    background:var(--panel2) !important;
  }
  .stSlider [data-testid="stTickBar"]{display:none;}
  div[data-testid="stSliderThumbValue"]{
    color:var(--safe) !important; font-family:'JetBrains Mono',monospace !important;
    font-weight:800 !important; font-size:13px !important;
  }
  .stSlider [role="slider"]{box-shadow:0 0 12px rgba(53,224,200,.75) !important;}

  /* ---------------- button ---------------- */
  .stButton > button[kind="primary"]{
    background:linear-gradient(92deg,var(--safe),#4BC9FF); color:#04121A;
    border:none; border-radius:12px; font-family:'Space Grotesk',sans-serif;
    font-weight:700; font-size:16px; letter-spacing:.03em; padding:15px 0;
    box-shadow:0 0 30px rgba(53,224,200,.28);
  }
  .stButton > button[kind="primary"]:hover{
    box-shadow:0 0 44px rgba(53,224,200,.5); transform:translateY(-1px);
  }

  /* ---------------- sidebar ---------------- */
  section[data-testid="stSidebar"]{
    background:var(--panel2); border-right:1px solid var(--line);
  }
  section[data-testid="stSidebar"] *{color:var(--text) !important;}
  section[data-testid="stSidebar"] .stButton > button{
    background:rgba(53,224,200,.05); color:var(--text) !important;
    border:1px solid var(--line2); border-radius:11px; font-weight:600; font-size:13.5px;
  }
  section[data-testid="stSidebar"] .stButton > button:hover{
    border-color:var(--safe); background:rgba(53,224,200,.12);
  }
  section[data-testid="stSidebar"] div[data-testid="stMetricValue"]{
    font-family:'JetBrains Mono',monospace !important; font-size:26px !important;
    font-weight:800 !important; color:var(--safe) !important;
  }
  section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] p{
    font-family:'JetBrains Mono',monospace !important; font-size:10px !important;
    letter-spacing:.16em; text-transform:uppercase; color:var(--muted) !important;
  }

  /* ---------------- verdict / gauge ---------------- */
  .verdict{
    background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line2); border-radius:18px;
    padding:22px 24px 20px; text-align:center; margin-bottom:14px;
  }
  .verdict .band{
    display:inline-block; font-family:'JetBrains Mono',monospace; font-size:11px;
    font-weight:800; letter-spacing:.2em; text-transform:uppercase;
    padding:7px 14px; border-radius:20px; margin-bottom:6px;
  }
  .gaugewrap{position:relative; margin:2px auto 0; width:270px;}
  .gaugenum{
    position:absolute; left:0; right:0; top:74px;
    font-family:'JetBrains Mono',monospace; font-size:52px; font-weight:800;
    letter-spacing:-.03em; line-height:1; color:#fff;
  }
  .gaugesub{
    position:absolute; left:0; right:0; top:132px;
    font-size:11.5px; color:var(--muted); letter-spacing:.06em;
  }
  .gaugeends{
    display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace;
    font-size:10px; color:var(--muted); margin-top:-6px; padding:0 10px;
  }

  /* ---------------- factors ---------------- */
  .factor{
    background:var(--panel2); border:1px solid var(--line);
    border-left:3px solid var(--line2); border-radius:12px;
    padding:13px 16px; margin-bottom:9px;
  }
  .factor.up{border-left-color:var(--risk); box-shadow:-8px 0 22px -14px var(--risk);}
  .factor.down{border-left-color:var(--safe); box-shadow:-8px 0 22px -14px var(--safe);}
  .factor .f-title{
    font-family:'Space Grotesk',sans-serif; font-size:14.5px; font-weight:600;
    color:#fff; line-height:1.3;
  }
  .factor.up .f-title span{color:var(--risk);}
  .factor.down .f-title span{color:var(--safe);}
  .factor .f-why{font-size:12.5px; color:var(--mid); margin-top:4px; line-height:1.55;}

  .action{
    background:linear-gradient(135deg, rgba(124,92,255,.14), rgba(124,92,255,.04));
    border:1px solid rgba(124,92,255,.35); border-radius:14px;
    padding:16px 18px; font-size:13.5px; color:var(--text); line-height:1.65;
  }
  .action b{color:#BCA8FF;}
  .action .tag{
    display:block; font-family:'JetBrains Mono',monospace; font-size:10px;
    letter-spacing:.2em; text-transform:uppercase; color:var(--violet);
    margin-bottom:7px; font-weight:700;
  }

  .credit{
    font-size:11.5px; color:var(--muted); border-top:1px solid var(--line);
    padding-top:16px; margin-top:30px; line-height:1.85;
  }
  .credit b{color:var(--mid);}

  div[data-testid="stExpander"]{
    border:1px solid var(--line) !important; border-radius:12px !important;
    background:var(--panel2) !important;
  }
  .stDataFrame{border-radius:10px; overflow:hidden;}

  /* ---------------- animations ---------------- */
  @keyframes rise{from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;}}
  @keyframes slidein{from{opacity:0; transform:translateX(-14px);} to{opacity:1; transform:none;}}
  @keyframes pulseband{
    0%,100%{box-shadow:0 0 0 0 rgba(255,255,255,0);}
    50%{box-shadow:0 0 22px 2px currentColor;}
  }
  @keyframes glowsweep{
    0%{background-position:-200% 0;} 100%{background-position:200% 0;}
  }
  .verdict{animation:rise .55s cubic-bezier(.22,1,.36,1) both;}
  .verdict .band{animation:pulseband 2.6s ease-in-out infinite;}
  .action{animation:rise .55s .22s cubic-bezier(.22,1,.36,1) both;}
  .factor{animation:slidein .5s cubic-bezier(.22,1,.36,1) both;}
  .hero h1 em{
    background-size:220% auto; animation:glowsweep 7s linear infinite;
  }
  .stButton > button[kind="primary"]{transition:box-shadow .25s, transform .18s;}
  .factor{transition:transform .18s, border-color .18s;}
  .factor:hover{transform:translateX(3px);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Telco Retention Engine · Logistic Regression</div>
      <h1>Churn <em>Risk Console</em></h1>
      <p>Score a customer's likelihood of cancelling, see which factors drove the
      call, and get a recommended retention action.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Loads the fitted Pipeline once and caches it across reruns."""
    return joblib.load(MODEL_PATH)


try:
    model = load_pipeline()
except FileNotFoundError:
    st.error(
        "Model file not found. Run `python main.py` once to train and save the "
        "pipeline, then reload this page."
    )
    st.stop()


# --------------------------------------------------------------------------
# demo presets -- makes live demos fast
# --------------------------------------------------------------------------
PRESETS = {
    "At-risk profile": dict(
        gender="Female", SeniorCitizen="No", Partner="No", Dependents="No",
        tenure=2, PhoneService="Yes", MultipleLines="No",
        InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
        StreamingMovies="Yes", Contract="Month-to-month", PaperlessBilling="Yes",
        PaymentMethod="Electronic check", MonthlyCharges=95.7,
    ),
    "Loyal profile": dict(
        gender="Male", SeniorCitizen="No", Partner="Yes", Dependents="Yes",
        tenure=64, PhoneService="Yes", MultipleLines="Yes",
        InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
        DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
        StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
        PaymentMethod="Bank transfer (automatic)", MonthlyCharges=61.0,
    ),
}

if "preset" not in st.session_state:
    st.session_state.preset = PRESETS["At-risk profile"].copy()

with st.sidebar:
    st.markdown('<div class="sectionlabel">Demo presets</div>', unsafe_allow_html=True)
    st.caption("Load a profile, then press Score customer.")
    for name in PRESETS:
        if st.button(name, width="stretch"):
            st.session_state.preset = PRESETS[name].copy()
            st.rerun()

    st.markdown('<div class="sectionlabel">Model</div>', unsafe_allow_html=True)
    st.caption(
        "Class-weighted Logistic Regression, chosen by cross-validated F1 over "
        "Random Forest and Gradient Boosting."
    )
    st.metric("Recall (churn)", "0.80")
    st.metric("ROC-AUC", "0.845")

p = st.session_state.preset


def idx(options, value):
    """Safe index lookup for preset defaults."""
    return options.index(value) if value in options else 0


# --------------------------------------------------------------------------
# input form
# --------------------------------------------------------------------------
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown('<div class="sectionlabel">Account</div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72, int(p["tenure"]))
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"],
        index=idx(["Month-to-month", "One year", "Two year"], p["Contract"]),
    )
    monthly = st.number_input(
        "Monthly charges ($)", 15.0, 130.0, float(p["MonthlyCharges"]), step=0.5
    )
    paperless = st.selectbox(
        "Paperless billing", ["Yes", "No"], index=idx(["Yes", "No"], p["PaperlessBilling"])
    )
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        index=idx(["Electronic check", "Mailed check",
                   "Bank transfer (automatic)", "Credit card (automatic)"],
                  p["PaymentMethod"]),
    )

with c2:
    st.markdown('<div class="sectionlabel">Services</div>', unsafe_allow_html=True)
    internet = st.selectbox(
        "Internet service", ["DSL", "Fiber optic", "No"],
        index=idx(["DSL", "Fiber optic", "No"], p["InternetService"]),
    )
    phone = st.selectbox(
        "Phone service", ["Yes", "No"], index=idx(["Yes", "No"], p["PhoneService"])
    )
    multi_opts = ["No", "Yes", "No phone service"]
    multiline = st.selectbox(
        "Multiple lines", multi_opts, index=idx(multi_opts, p["MultipleLines"])
    )
    addon_opts = ["No", "Yes", "No internet service"]
    security = st.selectbox(
        "Online security", addon_opts, index=idx(addon_opts, p["OnlineSecurity"])
    )
    support = st.selectbox(
        "Tech support", addon_opts, index=idx(addon_opts, p["TechSupport"])
    )

with c3:
    st.markdown('<div class="sectionlabel">Add-ons & profile</div>', unsafe_allow_html=True)
    backup = st.selectbox(
        "Online backup", addon_opts, index=idx(addon_opts, p["OnlineBackup"])
    )
    protection = st.selectbox(
        "Device protection", addon_opts, index=idx(addon_opts, p["DeviceProtection"])
    )
    tv = st.selectbox("Streaming TV", addon_opts, index=idx(addon_opts, p["StreamingTV"]))
    movies = st.selectbox(
        "Streaming movies", addon_opts, index=idx(addon_opts, p["StreamingMovies"])
    )
    gender = st.selectbox("Gender", ["Female", "Male"], index=idx(["Female", "Male"], p["gender"]))
    senior = st.selectbox("Senior citizen", ["No", "Yes"], index=idx(["No", "Yes"], p["SeniorCitizen"]))
    partner = st.selectbox("Partner", ["No", "Yes"], index=idx(["No", "Yes"], p["Partner"]))
    dependents = st.selectbox("Dependents", ["No", "Yes"], index=idx(["No", "Yes"], p["Dependents"]))

# TotalCharges is a lifetime figure, so derive it rather than asking for it.
total_charges = round(monthly * tenure, 2)

st.write("")
go = st.button("Score customer", type="primary", width="stretch")


# --------------------------------------------------------------------------
# explanation helper
# --------------------------------------------------------------------------
def risk_factors(row: dict) -> list:
    """
    Plain-English drivers, ordered by the model's own top coefficients
    (contract type, internet service, tenure, monthly charges, add-ons).
    Returns (direction, title, why) tuples.
    """
    out = []
    if row["Contract"] == "Month-to-month":
        out.append(("up", "Month-to-month contract",
                    "The single strongest churn signal — no commitment to stay. "
                    "In the data these customers churn ~43% vs ~3% on two-year plans."))
    elif row["Contract"] == "Two year":
        out.append(("down", "Two-year contract",
                    "The strongest retention signal in the model."))
    else:
        out.append(("down", "One-year contract",
                    "Some commitment — churns far less than month-to-month."))

    if row["tenure"] <= 6:
        out.append(("up", f"New customer ({row['tenure']} months)",
                    "Churn is concentrated in the first months — under-6-month "
                    "customers churn at roughly 53%."))
    elif row["tenure"] >= 48:
        out.append(("down", f"Long tenure ({row['tenure']} months)",
                    "Established customers rarely leave."))

    if row["InternetService"] == "Fiber optic":
        out.append(("up", "Fiber optic internet",
                    "Associated with higher churn — typically higher bills and "
                    "more price sensitivity."))
    elif row["InternetService"] == "No":
        out.append(("down", "No internet service",
                    "Phone-only customers churn less."))

    if row["MonthlyCharges"] >= 80:
        out.append(("up", f"High monthly bill (${row['MonthlyCharges']:.0f})",
                    "Higher charges increase price-driven churn."))
    elif row["MonthlyCharges"] <= 45:
        out.append(("down", f"Low monthly bill (${row['MonthlyCharges']:.0f})",
                    "Low bills reduce price pressure."))

    protective = [row["OnlineSecurity"], row["TechSupport"],
                  row["OnlineBackup"], row["DeviceProtection"]]
    have = sum(1 for v in protective if v == "Yes")
    if have == 0 and row["InternetService"] != "No":
        out.append(("up", "No support or security add-ons",
                    "Customers without add-ons are less 'locked in' and churn more."))
    elif have >= 3:
        out.append(("down", f"{have} protective add-ons",
                    "Bundled services make an account stickier."))

    if row["PaymentMethod"] == "Electronic check":
        out.append(("up", "Pays by electronic check",
                    "This payment method is linked to higher churn; automatic "
                    "payment methods correlate with retention."))
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
if go:
    row = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiline, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total_charges,
    }

    # add_features() must run first -- the pipeline expects the engineered columns.
    X = add_features(pd.DataFrame([row]))
    proba = float(model.predict_proba(X)[:, 1][0])
    flagged = proba >= 0.50

    if proba >= 0.70:
        band, colour, glow = "High risk", "#FF4D6D", "rgba(255,77,109,.22)"
    elif flagged:
        band, colour, glow = "Elevated risk", "#FFB13D", "rgba(255,177,61,.20)"
    else:
        band, colour, glow = "Low risk", "#35E0C8", "rgba(53,224,200,.18)"

    ARC = 314.159  # length of a 180-degree arc at r=100
    dash = ARC * (1 - proba)

    st.write("")
    left, right = st.columns([1, 1.15], gap="medium")

    with left:
        pct = int(round(proba * 100))
        st.markdown(
            f"""
            <style>
              @property --gn{{syntax:'<integer>'; initial-value:0; inherits:false;}}
              @keyframes sweep{pct}{{
                from{{stroke-dashoffset:{ARC:.1f};}} to{{stroke-dashoffset:{dash:.1f};}}
              }}
              @keyframes tick{pct}{{ to{{--gn:{pct};}} }}
              .arcv{pct}{{
                stroke-dashoffset:{ARC:.1f};
                animation:sweep{pct} 1.5s .15s cubic-bezier(.34,1.4,.64,1) forwards;
              }}
              .cn{pct}{{
                counter-reset:gn var(--gn);
                animation:tick{pct} 1.5s .15s cubic-bezier(.34,1.4,.64,1) forwards;
              }}
              .cn{pct}:after{{content:counter(gn);}}
            </style>
            <div class="verdict">
              <div class="band" style="color:{colour};background:{glow};
                   border:1px solid {colour}55">{band}</div>
              <div class="gaugewrap">
                <svg viewBox="0 0 270 150" width="270" height="150">
                  <defs>
                    <linearGradient id="gg" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stop-color="#35E0C8"/>
                      <stop offset="52%" stop-color="#FFB13D"/>
                      <stop offset="100%" stop-color="#FF4D6D"/>
                    </linearGradient>
                    <filter id="gl" x="-60%" y="-60%" width="220%" height="220%">
                      <feGaussianBlur stdDeviation="7" result="b"/>
                      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                  </defs>
                  <path d="M 35 130 A 100 100 0 0 1 235 130" fill="none"
                        stroke="#1E2A45" stroke-width="15" stroke-linecap="round"/>
                  <path class="arcv{pct}" d="M 35 130 A 100 100 0 0 1 235 130" fill="none"
                        stroke="url(#gg)" stroke-width="15" stroke-linecap="round"
                        stroke-dasharray="{ARC:.1f}" filter="url(#gl)"/>
                </svg>
                <div class="gaugenum" style="text-shadow:0 0 26px {colour}">
                  <span class="cn{pct}"></span><span style="font-size:26px">%</span>
                </div>
                <div class="gaugesub">probability of churn</div>
              </div>
              <div class="gaugeends"><span>0 · STAYS</span><span>LEAVES · 100</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if flagged:
            st.markdown(
                '<div class="action"><span class="tag">Recommended action</span><b>Add to the retention queue.</b> '
                'The highest-leverage offer is a discounted <b>one- or two-year '
                'contract</b>, since contract length is the model\'s strongest driver. '
                'Bundling tech support or online security also reduces churn risk.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="action"><span class="tag">Recommended action</span><b>No intervention needed.</b> '
                'Keep in routine monitoring — re-score if the contract lapses to '
                'month-to-month or the monthly bill rises sharply.</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="sectionlabel">Why the model called it this way</div>',
                    unsafe_allow_html=True)
        for i, (direction, title, why) in enumerate(risk_factors(row)):
            arrow = "▲" if direction == "up" else "▼"
            st.markdown(
                f'<div class="factor {direction}" style="animation-delay:{0.07*i+0.15:.2f}s">'
                f'<div class="f-title"><span>{arrow}</span> {title}</div>'
                f'<div class="f-why">{why}</div></div>',
                unsafe_allow_html=True,
            )

    with st.expander("Engineered features the pipeline computed for this customer"):
        st.caption(
            "These are created by add_features() before the pipeline runs — the same "
            "code used in training."
        )
        st.dataframe(
            X[["tenure_group", "num_services", "avg_charges_per_mo",
               "is_new_customer", "TotalCharges"]],
            width="stretch", hide_index=True,
        )

st.markdown(
    """
    <div class="credit">
      Trained on the public IBM Telco Customer Churn dataset (7,043 customers, ~27% churn).
      Test-set performance: recall 0.80, ROC-AUC 0.845. Recall is prioritised because
      missing a customer who leaves costs more than a false alarm.<br>
      Built by <b>Tanishak Agarwal</b> · github.com/Tanishak10789/customer-churn-prediction
    </div>
    """,
    unsafe_allow_html=True,
)