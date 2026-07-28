"""
UBL Workforce Intelligence — Codex-style chat UI.
Run:  streamlit run app.py
"""
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import pipeline as P

st.set_page_config(page_title="UBL Workforce Intelligence", page_icon=None, layout="wide")
plt.rcParams["figure.dpi"] = 100

# ── Cohere-inspired theme (chat mode) ────────────────────────────────────
_COHERE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@400;500;700&display=swap');

  :root {
    --ink: #212121;
    --muted: #93939f;
    --slate: #75758a;
    --hairline: #d9d9dd;
    --border-light: #e5e7eb;
    --canvas: #ffffff;
    --stone: #eeece7;
    --pale-green: #edfce9;
    --pale-blue: #f1f5ff;
    --near-black: #17171c;
    --coral: #ff7759;
    --soft-coral: #ffad9b;
    --action-blue: #1863dc;
    --deep-green: #003c33;
    --navy: #071829;
    --focus: #4c6ee6;
  }

  /* Sidebar visible (Power BI mode) — see dedicated sidebar block below. */
  [data-testid="stSidebarCollapsedControl"] { display: none !important; }

  html, body, [class*="css"] {
    font-family: 'Inter', 'Space Grotesk', system-ui, sans-serif !important;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
  }

  .stApp { background: var(--canvas); }
  [data-testid="stAppViewContainer"] { background: var(--canvas); }
  .main .block-container {
    max-width: 760px !important;
    padding-top: 80px !important;
    padding-bottom: 200px !important;
  }

  h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--near-black);
    font-weight: 500;
  }
  h1 { font-size: 56px !important; line-height: 1.05 !important; letter-spacing: -1.6px !important; text-align: center; }
  h2 { font-size: 22px !important; line-height: 1.3 !important; text-align: center; font-weight: 400 !important; color: var(--muted) !important; }
  h3 { font-size: 20px !important; line-height: 1.3 !important; }
  h4 { font-size: 18px !important; }

  .hero-title { text-align: center; margin-top: 8vh; margin-bottom: 6px; }
  .hero-sub { text-align: center; color: var(--muted); font-size: 15px; margin-bottom: 28px; }

  /* Dashboard filename — large, vertically aligned with back arrow */
  .dash-filename {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 32px !important;
    font-weight: 500 !important;
    letter-spacing: -0.72px !important;
    color: var(--near-black) !important;
    line-height: 1.1 !important;
    padding: 6px 0 2px 0 !important;
  }
  /* Stats line — small, muted, inline below filename */
  .dash-stats {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    color: var(--muted) !important;
    line-height: 1.4 !important;
    padding: 0 0 6px 0 !important;
  }

  /* Analysis dropdown — pill style */
  div[data-testid="stSelectbox"] > div > div {
    background: var(--canvas) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 24px !important;
    padding: 6px 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
  }
  div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--near-black) !important;
    box-shadow: 0 0 0 3px rgba(23,23,28,0.08) !important;
  }

  .chat-msg { padding: 18px 0; border-bottom: 1px solid var(--border-light); }
  .chat-user { font-weight: 500; color: var(--near-black); }
  .chat-asst { color: var(--ink); }
  .chat-meta { color: var(--muted); font-size: 12px; font-family: 'Space Grotesk', monospace; letter-spacing: 0.28px; text-transform: uppercase; margin-bottom: 6px; }

  .composer-wrap {
    position: fixed; left: 0; right: 0; bottom: 0;
    background: linear-gradient(to top, var(--canvas) 70%, rgba(255,255,255,0));
    padding: 24px 16px 28px; z-index: 999;
  }
  .composer-inner { max-width: 760px; margin: 0 auto; }

  .stChatInput > div {
    background: var(--canvas) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 28px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    padding: 4px 8px !important;
  }
  .stChatInput > div:focus-within {
    border-color: var(--near-black) !important;
    box-shadow: 0 0 0 3px rgba(23,23,28,0.08) !important;
  }
  .stChatInput textarea {
    background: transparent !important;
    border: none !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 8px !important;
  }
  .stChatInput textarea::placeholder { color: var(--muted) !important; }

  /* File uploader — boxed, label prominent, compact */
  [data-testid="stFileUploader"] {
    background: var(--canvas) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    transition: border-color .15s ease;
  }
  [data-testid="stFileUploader"]:hover { border-color: var(--near-black) !important; }

  [data-testid="stFileUploaderDropzone"] {
    background: var(--stone) !important;
    border: 1px dashed var(--hairline) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
  }
  [data-testid="stFileUploaderDropzone"] section { padding: 0 !important; }
  [data-testid="stFileUploaderDropzone"] button { padding: 4px 10px !important; font-size: 12px !important; }

  /* Make the uploader label the box title */
  [data-testid="stFileUploader"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.28px !important;
    color: var(--near-black) !important;
    margin-bottom: 8px !important;
  }

  .stButton > button {
    background: var(--near-black) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Bare back arrow — no pill, no fill */
  div[data-testid="stButton"]:has(button[key="back_arrow"]) button {
    background: transparent !important;
    color: var(--near-black) !important;
    border: none !important;
    padding: 4px 8px !important;
    font-size: 28px !important;
    line-height: 1 !important;
    border-radius: 8px !important;
  }
  div[data-testid="stButton"]:has(button[key="back_arrow"]) button:hover {
    background: var(--stone) !important;
    opacity: 1 !important;
  }

  .stDataFrame, [data-testid="stDataFrame"] {
    border: none !important;
    box-shadow: none !important;
    border-radius: 8px !important;
  }
  .stDataFrame table { border-collapse: collapse !important; }
  .stDataFrame th {
    background: var(--canvas) !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    letter-spacing: 0.28px !important;
    font-family: 'Space Grotesk', monospace !important;
    border-bottom: 1px solid var(--hairline) !important;
  }
  .stDataFrame td { border-bottom: 1px solid var(--border-light) !important; font-size: 14px !important; }

  [data-testid="stMetric"] { background: transparent !important; padding: 0 !important; border: none !important; }
  [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 32px !important;
    font-weight: 500 !important;
    color: var(--near-black) !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 11px !important; text-transform: uppercase !important;
    letter-spacing: 0.28px !important; color: var(--muted) !important;
    font-family: 'Space Grotesk', monospace !important;
  }

  .stAlert { border-radius: 10px !important; border-left: 3px solid !important; }
  hr { border-color: var(--hairline) !important; }
  .stCaption, small { color: var(--muted) !important; font-size: 12px !important; }
  a { color: var(--action-blue) !important; text-decoration: underline; text-underline-offset: 3px; }
  code, pre {
    background: var(--stone) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', monospace !important;
    font-size: 13px !important;
  }

  #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
  .viewerBadge_link__qRIco { display: none !important; }
  [data-testid="stDecoration"] { display: none; }
  [data-testid="stHeader"] { background: transparent; }

  /* ── Top navbar (UBL placeholder) ──────────────────────────────────── */
  div.topnav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 52px;
    padding: 0 32px;
    border-bottom: 1px solid var(--hairline);
    background: var(--canvas);
    margin: -80px -16px 32px -16px;
  }
  div.topnav .topnav-left,
  div.topnav .topnav-right {
    display: flex; align-items: center; gap: 10px;
  }
  div.topnav .topnav-mark {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.5px;
    color: var(--near-black);
  }
  div.topnav .topnav-sep {
    color: var(--hairline);
    font-size: 14px;
  }
  div.topnav .topnav-section {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 14px;
    color: var(--near-black);
  }
  div.topnav .topnav-meta {
    font-family: 'Space Grotesk', monospace;
    font-size: 11px;
    letter-spacing: 0.28px;
    text-transform: uppercase;
    color: var(--muted);
  }

  /* ── Power BI-style sidebar (re-enabled) ────────────────────────────── */
  [data-testid="stSidebar"] {
    display: block !important;
    background: var(--canvas) !important;
    border-right: 1px solid var(--hairline) !important;
    width: 280px !important;
    min-width: 280px !important;
    padding: 16px 12px !important;
  }
  [data-testid="stSidebar"] [data-testid="stExpander"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
  }
  [data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.28px !important;
    font-weight: 500 !important;
    color: var(--near-black) !important;
    padding: 8px 0 !important;
  }

  /* ── Slicer chip row ───────────────────────────────────────────────── */
  div.slicer-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin: 12px 0 4px 0;
    align-items: center;
  }
  div.slicer-row .chip {
    display: inline-block;
    background: var(--stone);
    border: 1px solid var(--hairline);
    border-radius: 16px;
    padding: 4px 10px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: var(--near-black);
  }
  div.slicer-row .chip-meta {
    margin-left: auto;
    font-family: 'Space Grotesk', monospace;
    font-size: 11px;
    letter-spacing: 0.28px;
    text-transform: uppercase;
    color: var(--muted);
  }

  /* ── Animations (Cohere-style: fast, restrained) ───────────────────── */
  @keyframes fade-up {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes grow-x {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
  }
  @keyframes shimmer {
    0%   { background-position: -200px 0; }
    100% { background-position: 200px 0; }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.5; }
  }

  /* Fade-up on load — hero, navbar, dashboard header */
  .anim-fade-up { animation: fade-up 220ms ease-out both; }
  .anim-fade-up-1 { animation: fade-up 220ms ease-out 60ms both; }
  .anim-fade-up-2 { animation: fade-up 220ms ease-out 120ms both; }

  /* Section reveal — content slides in under header */
  .anim-section { animation: fade-up 260ms ease-out both; }

  /* Divider grow-in */
  [data-testid="stDivider"] {
    transform-origin: left;
    animation: grow-x 380ms ease-out both;
  }

  /* Skeleton shimmer on upload boxes before file lands */
  [data-testid="stFileUploader"]:not(:has(input[type="file"]:valid)) {
    background: linear-gradient(90deg, var(--canvas) 0%, var(--stone) 50%, var(--canvas) 100%);
    background-size: 400px 100%;
    animation: shimmer 1800ms ease-in-out infinite;
  }

  /* Back arrow nudge on hover */
  div[data-testid="stButton"]:has(button[key="back_arrow"]) button:hover {
    transform: translateX(-2px);
    transition: transform 120ms ease-out;
  }

  /* Primary button press feedback */
  .stButton > button:active {
    transform: scale(0.98);
    transition: transform 80ms ease-out;
  }

  /* Dropdown open — subtle lift */
  div[data-testid="stSelectbox"] > div > div {
    transition: border-color 150ms ease-out, box-shadow 150ms ease-out, transform 150ms ease-out;
  }
  div[data-testid="stSelectbox"] > div > div:hover {
    transform: translateY(-1px);
  }
</style>
"""
st.markdown(_COHERE_CSS, unsafe_allow_html=True)

BAND_COLORS = {"High Performer": "#003c33", "Solid": "#4c9f70", "Average": "#d4a017",
               "Below Average": "#d9822b", "Low Performer": "#c0392b"}

# ── session state ───────────────────────────────────────────────────────
for k, v in {"messages": [], "uploaded_data_name": None,
             "uploaded_hier_name": None, "uploaded_at": None,
             "uploaded_size": None, "active_tile_label": "— Select analysis —",
             "open_section": None, "pipeline_result": None,
             "pipeline_error": None, "flt_jobs": [], "flt_regions": [],
             "flt_bands": [], "flt_expanders": {"filters": True, "viz": True, "fields": True}
             }.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── top navbar (persistent) ─────────────────────────────────────────────
st.markdown("""
<div class="topnav anim-fade-up">
  <div class="topnav-left">
    <span class="topnav-mark">UBL</span>
    <span class="topnav-sep">·</span>
    <span class="topnav-section">Workforce Intelligence</span>
  </div>
  <div class="topnav-right">
    <span class="topnav-meta">Operations · Internal</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── pipeline runner (cached, runs once per file+params) ─────────────────
@st.cache_data(show_spinner=False)
def _run_pipeline(data_bytes, data_name, hier_bytes, hier_name):
    buf = io.BytesIO(data_bytes); buf.name = data_name
    hdf = None
    if hier_bytes is not None:
        hbuf = io.BytesIO(hier_bytes)
        hdf = pd.read_csv(hbuf) if hier_name.lower().endswith(".csv") else pd.read_excel(hbuf)
    emp, meta, review, skipped = P.load_and_prepare(buf, hdf)
    emp, dir_tbl = P.score_employees(emp, w_ops=0.80)
    sur = P.surrogate_model(emp)
    hp = P.hp_engine(emp, threshold=80)
    lp = P.lp_engine(emp, threshold=40)
    gr = P.growth_engine(emp)
    emp = gr["emp"]
    pool, trait_res, traits, trait_missing = P.trait_personas(emp)
    fair_tabs, fair_checks, fair_failed = P.fairness_audit(emp, sur["model_score"])
    return dict(emp=emp, meta=meta, review=review, skipped=skipped, dir_tbl=dir_tbl,
                sur=sur, hp=hp, lp=lp, gr=gr, pool=pool, trait_res=trait_res,
                traits=traits, trait_missing=trait_missing,
                fair_tabs=fair_tabs, fair_checks=fair_checks, fair_failed=fair_failed)


# ── hero + uploaders — only visible before a file is loaded ───────────
if st.session_state.pipeline_result is None and st.session_state.pipeline_error is None:
    st.markdown("<div class='hero-title anim-fade-up'><h1>Workforce Intelligence</h1></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='hero-sub anim-fade-up-1'>Attach an employee extract to begin. "
                "Every figure recomputes from the uploaded file.</div>",
                unsafe_allow_html=True)
    box_l, box_r = st.columns(2)
    with box_l:
        data_file = st.file_uploader("Attach data", type=["xlsx", "xls", "csv"],
                                     key="data_attach")
    with box_r:
        hier_file = st.file_uploader("Attach hierarchy (optional)",
                                     type=["xlsx", "xls", "csv"],
                                     key="hier_attach")
else:
    data_file = None
    hier_file = None

# ── run pipeline whenever a new data file lands ─────────────────────────
if data_file is not None and (
    st.session_state.uploaded_data_name != data_file.name or st.session_state.pipeline_result is None
):
    try:
        with st.spinner("Loading and scoring the extract …"):
            R = _run_pipeline(
                data_file.getvalue(), data_file.name,
                hier_file.getvalue() if hier_file else None,
                hier_file.name if hier_file else "",
            )
        st.session_state.pipeline_result = R
        st.session_state.pipeline_error = None
        st.session_state.uploaded_data_name = data_file.name
        st.session_state.uploaded_size = len(data_file.getvalue())
        st.session_state.uploaded_at = __import__("datetime").datetime.now()
        if hier_file:
            st.session_state.uploaded_hier_name = hier_file.name
        st.session_state.messages = [
            {"role": "assistant", "content":
             f"Loaded `{data_file.name}` — {R['meta']['rows']:,} rows, "
             f"{R['meta']['n_emp']:,} employees, {R['meta']['network_days']} business days. "
             f"Ask me anything about the workforce."}
        ]
    except Exception as e:
        st.session_state.pipeline_error = e
        st.session_state.pipeline_result = None

R = st.session_state.pipeline_result
err = st.session_state.pipeline_error


# ── filter helper: apply sidebar filters to a view of the data ─────────
def filtered_emp(R):
    emp = R["emp"]
    if st.session_state.flt_jobs:
        emp = emp[emp["job_group"].isin(st.session_state.flt_jobs)]
    if st.session_state.flt_regions and "region" in emp.columns:
        emp = emp[emp["region"].isin(st.session_state.flt_regions)]
    if st.session_state.flt_bands:
        emp = emp[emp["band"].isin(st.session_state.flt_bands)]
    return emp


# ── tile dispatcher: key → pipeline call ───────────────────────────────
def answer(key: str):
    """Map a tile key to a structured response payload."""
    R = st.session_state.pipeline_result
    if R is None:
        return {"content": "Attach an employee extract first."}

    emp = R["emp"]; meta = R["meta"]; sur = R["sur"]
    hp = R["hp"]; lp = R["lp"]; gr = R["gr"]
    fair_checks = R["fair_checks"]; fair_failed = R["fair_failed"]

    if key == "headcount":
        return {"content": f"**{meta['n_emp']:,}** employees across "
                           f"**{emp['branch_code'].nunique()}** branches in a "
                           f"**{meta['network_days']}**-business-day window.",
                "metrics": [("Employees", f"{meta['n_emp']:,}"),
                            ("Branches", f"{emp['branch_code'].nunique():,}"),
                            ("Window days", f"{meta['network_days']}"),
                            ("Median score", f"{emp['performance_score'].median():.1f}")]}

    if key == "bands":
        bc = emp["band"].value_counts().reindex(list(BAND_COLORS)).fillna(0).astype(int)
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar(bc.index, bc.values, color=[BAND_COLORS[b] for b in bc.index])
        for i, v in enumerate(bc.values):
            ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        return {"content": "Median score **{:.1f}**. Distribution by band:".format(
                    emp["performance_score"].median()),
                "figures": [("Band counts", fig)]}

    if key == "top":
        view = emp.sort_values("performance_score", ascending=False).head(15)
        df = view[["EMPLOYEE_NUMBER", "job_group", "Grade", "branch_code", "City", "region",
                   "performance_score", "band", "decile"]].round(2)
        return {"content": "Top 15 by composite score.",
                "dataframes": [("", df)]}

    if key == "low":
        view = emp.sort_values("performance_score").head(15)
        df = view[["EMPLOYEE_NUMBER", "job_group", "Grade", "branch_code", "City", "region",
                   "performance_score", "band", "decile"]].round(2)
        return {"content": "Bottom 15 by composite score. **Governance:** environmental factors "
                           "(branch health, team size, benchmark) are mandatory inputs to the LP "
                           "segmentation; do not attribute low performance to individuals alone.",
                "dataframes": [("", df)]}

    if key == "fairness":
        head = ("**Fairness audit FAILED.** Scores must be withheld from consequential HR use "
                "pending governance review." if fair_failed else
                "**Fairness audit PASSED.** No disparity exceeded thresholds.")
        return {"content": head, "dataframes": [("", fair_checks)]}

    if key == "shap":
        fig = plt.figure(figsize=(7, 3.6))
        shap.plots.beeswarm(sur["shap"], max_display=10, show=False)
        return {"content": f"Surrogate model test R² **{sur['r2']:.3f}** — top 10 global drivers:",
                "figures": [("", fig)]}

    if key == "hp_personas":
        prof = hp["profile"].copy()
        prof.index = [hp["personas"][c] for c in prof.index]
        return {"content": f"**{len(hp['hp'])}** high performers split into **{hp['k']}** personas.",
                "dataframes": [("", prof)]}

    if key == "lp_segments":
        return {"content": "Low-performer segments with environment-aware diagnoses.",
                "dataframes": [("Profile", lp["profile"]),
                               ("Diagnoses", lp["diagnoses"])]}

    if key == "growth":
        if gr["parity_gap_pp"] < 5:
            parity = (f"Growth-driver gender mix mirrors the workforce "
                      f"(max gap {gr['parity_gap_pp']:.1f}pp < 5pp).")
        else:
            parity = (f"Growth-driver gender mix diverges from workforce by "
                      f"{gr['parity_gap_pp']:.1f}pp — investigate structural causes.")
        return {"content": parity, "dataframes": [("Top-6 driver lifts", gr["lifts"])]}

    if key == "hiring":
        recs = P.hiring_recs(emp, R["pool"], "region", 0.12, 0.05, 0.03)
        return {"content": "Net hires per region at attrition 12%, growth 5%, promotions 3%.",
                "dataframes": [("", recs)]}

    if key == "rollup":
        roll = P.rollup(emp, "region")
        return {"content": "Hierarchy rollup at **region**. Statistically stable cells only (n ≥ 8).",
                "dataframes": [("", roll)]}

    if key == "deciles":
        job = emp["job_group"].value_counts().idxmax()
        g = emp[emp["job_group"] == job]
        order = [f"D{i}" for i in range(1, 11)]
        t = g.groupby("decile")[["trxn_count_per_day", "trxn_amount_m_per_day",
                                  "avg_ticket_m", "active_days_ratio"]].mean().reindex(order).round(2)
        return {"content": f"Per-job decile averages for **{job}** (D1 = top 10%).",
                "dataframes": [("", t)]}

    return {"content": "Unknown section."}


# ── helper: render an analysis result payload ─────────────────────────
def render_payload(msg):
    body = msg.get("content")
    if body:
        st.markdown(body)
    for label, value in msg.get("metrics", []):
        pass  # rendered below as a tile-wide metric row
    metrics = msg.get("metrics", [])
    if metrics:
        cols = st.columns(len(metrics))
        for c, (label, value) in zip(cols, metrics):
            c.metric(label, value)
    for caption, df in msg.get("dataframes", []):
        if caption:
            st.caption(caption)
        st.dataframe(df, use_container_width=True, hide_index=True)
    for caption, fig in msg.get("figures", []):
        if caption:
            st.caption(caption)
        st.pyplot(fig, clear_figure=True)


# ── dashboard view (only renders when data is loaded) ─────────────────
if err is not None:
    st.error("Pipeline error — the uploaded file may have missing or mismatched columns.")
    with st.expander("Error details"):
        st.code(f"{type(err).__name__}: {err}", language="text")
    st.markdown("Expected columns include **`Employee`**, **`Trxn_Date`**, "
                "transaction KPIs, and branch/HR attributes.")
elif R is not None:
    emp_full = R["emp"]; meta = R["meta"]

    # ── sidebar: Filters / Visualizations / Fields ────────────────────
    with st.sidebar:
        st.markdown("**Filters**")
        with st.expander("Filters", expanded=st.session_state.flt_expanders["filters"]):
            jobs = sorted(emp_full["job_group"].dropna().unique())
            regions = sorted(emp_full.get("region", pd.Series(dtype=object)).dropna().unique())
            bands = list(BAND_COLORS)
            st.session_state.flt_jobs = st.multiselect("Job", jobs,
                                                       default=st.session_state.flt_jobs)
            st.session_state.flt_regions = st.multiselect("Region", regions,
                                                          default=st.session_state.flt_regions)
            st.session_state.flt_bands = st.multiselect("Band", bands,
                                                        default=st.session_state.flt_bands)
            if st.button("Clear filters", use_container_width=True, key="clear_flt"):
                st.session_state.flt_jobs = []
                st.session_state.flt_regions = []
                st.session_state.flt_bands = []
                st.rerun()
        with st.expander("Visualizations", expanded=st.session_state.flt_expanders["viz"]):
            st.caption("Active visuals on the canvas:")
            st.markdown("- KPI strip\n- Band distribution\n- SHAP drivers\n"
                        "- Top performers\n- Decile profile\n- Hiring recs")
        with st.expander("Fields", expanded=st.session_state.flt_expanders["fields"]):
            for f in ["EMPLOYEE_NUMBER", "branch_code", "City", "region", "Grade",
                      "performance_score", "band", "decile", "growth_index"]:
                st.checkbox(f, value=False, key=f"fld_{f}", disabled=True,
                            help="Drag-to-pivot is not available in this view.")

    # ── filtered view ────────────────────────────────────────────────
    emp = filtered_emp(R)
    n_filtered = len(emp)
    is_filtered = bool(st.session_state.flt_jobs or st.session_state.flt_regions or st.session_state.flt_bands)

    # ── header row ───────────────────────────────────────────────────
    size_kb = (st.session_state.uploaded_size or 0) / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.0f} KB"
    n_total = meta["n_emp"]
    head_stats = (f"{n_total:,} employees · "
                  f"{meta['network_days']} business days · "
                  f"{emp_full['branch_code'].nunique()} branches · "
                  f"{size_str}")
    back_l, mid_l = st.columns([0.6, 11], vertical_alignment="center")
    with back_l:
        if st.button("←", key="back_arrow", help="Replace file"):
            for k in ("pipeline_result", "pipeline_error", "uploaded_data_name",
                      "uploaded_hier_name", "uploaded_at", "uploaded_size",
                      "messages", "open_section",
                      "flt_jobs", "flt_regions", "flt_bands"):
                st.session_state[k] = [] if k.startswith("flt_") else None
            st.rerun()
    with mid_l:
        st.markdown(
            f"<div class='dash-filename anim-fade-up'>{st.session_state.uploaded_data_name}</div>"
            f"<div class='dash-stats anim-fade-up-1'>{head_stats}</div>",
            unsafe_allow_html=True)

    # ── slicer chip row ──────────────────────────────────────────────
    chips = []
    for j in st.session_state.flt_jobs:    chips.append(("Job", j))
    for r in st.session_state.flt_regions: chips.append(("Region", r))
    for b in st.session_state.flt_bands:   chips.append(("Band", b))
    match_str = f"{n_filtered:,} of {n_total:,} match"
    if chips:
        st.markdown(
            "<div class='slicer-row'>"
            + "".join(f"<span class='chip'>{k}: {v}</span>" for k, v in chips)
            + f"<span class='chip-meta'>{match_str}</span>"
            + "</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='slicer-row'><span class='chip-meta'>"
                    f"{match_str} · no filters</span></div>", unsafe_allow_html=True)

    if meta["network_days"] < 60:
        st.warning(f"Window is only {meta['network_days']} business days — "
                   "treat scores as short-window throughput indicators.")
    if R["fair_failed"]:
        st.error("Fairness audit FAILED — scores must be withheld from consequential HR use.")

    st.divider()

    # ── 2x2 canvas ───────────────────────────────────────────────────
    st.markdown("<div class='anim-section'>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Employees (filtered)", f"{n_filtered:,}",
             delta=f"{n_filtered - n_total:,}" if is_filtered else None,
             delta_color="off")
    k2.metric("Branches", f"{emp['branch_code'].nunique() if n_filtered else 0}")
    k3.metric("Median score",
              f"{emp['performance_score'].median():.1f}" if n_filtered else "—")
    k4.metric("High performers",
              int((emp["band"] == "High Performer").sum()) if n_filtered else 0)

    st.write("")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**Band distribution**")
        if n_filtered:
            bc = emp["band"].value_counts().reindex(list(BAND_COLORS)).fillna(0).astype(int)
            fig, ax = plt.subplots(figsize=(5, 2.4))
            ax.bar(bc.index, bc.values, color=[BAND_COLORS[b] for b in bc.index])
            for i, v in enumerate(bc.values):
                ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig, clear_figure=True)
        else:
            st.caption("No data matches current filters.")
    with r1c2:
        st.markdown("**SHAP drivers**")
        if n_filtered >= 20:
            try:
                feats = [c for c in R["sur"]["features"] if c in emp.columns]
                if len(feats) >= 3:
                    from sklearn.model_selection import train_test_split
                    from xgboost import XGBRegressor
                    Xf = emp[feats].astype(float)
                    yf = emp["performance_score"].astype(float)
                    if yf.std() > 0:
                        Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.25, random_state=42)
                        m = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         subsample=0.9, colsample_bytree=0.9, random_state=42).fit(Xtr, ytr)
                        sv = shap.TreeExplainer(m)(Xf)
                        fig = plt.figure(figsize=(5, 2.6))
                        shap.plots.beeswarm(sv, max_display=8, show=False)
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.caption("Score variance is zero — SHAP unavailable.")
                else:
                    st.caption("Not enough features for SHAP.")
            except Exception as e:
                st.caption(f"SHAP unavailable on filtered set: {type(e).__name__}")
        else:
            st.caption("Need at least 20 rows for SHAP.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**Top performers**")
        if n_filtered:
            view = emp.sort_values("performance_score", ascending=False).head(10)
            st.dataframe(
                view[["EMPLOYEE_NUMBER", "job_group", "branch_code",
                      "performance_score", "band"]].round(2),
                use_container_width=True, hide_index=True, height=280)
        else:
            st.caption("No data matches current filters.")
    with r2c2:
        st.markdown("**Decile profile**")
        if n_filtered:
            job = emp["job_group"].value_counts().idxmax()
            g = emp[emp["job_group"] == job]
            order = [f"D{i}" for i in range(1, 11)]
            t = g.groupby("decile")[["trxn_count_per_day", "trxn_amount_m_per_day",
                                      "active_days_ratio"]].mean().reindex(order).round(2)
            st.dataframe(t, use_container_width=True, height=280)
            st.caption(f"Job: {job}")
        else:
            st.caption("No data matches current filters.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── detailed sections ────────────────────────────────────────────
    with st.expander("Fairness audit", expanded=False):
        st.dataframe(R["fair_checks"], use_container_width=True, hide_index=True)
    with st.expander("HP personas", expanded=False):
        prof = R["hp"]["profile"].copy()
        prof.index = [R["hp"]["personas"][c] for c in prof.index]
        st.dataframe(prof, use_container_width=True)
    with st.expander("LP segments", expanded=False):
        st.dataframe(R["lp"]["profile"], use_container_width=True)
        st.dataframe(R["lp"]["diagnoses"], use_container_width=True, hide_index=True)
    with st.expander("Growth drivers", expanded=False):
        st.dataframe(R["gr"]["lifts"], use_container_width=True, hide_index=True)
    with st.expander("Hiring recs (per region)", expanded=False):
        recs = P.hiring_recs(emp, R["pool"], "region", 0.12, 0.05, 0.03)
        st.dataframe(recs, use_container_width=True, hide_index=True)
    with st.expander("Hierarchy rollup (region)", expanded=False):
        st.dataframe(P.rollup(emp, "region"), use_container_width=True, height=320)
    with st.expander("Low performers", expanded=False):
        if n_filtered:
            view = emp.sort_values("performance_score").head(15)
            st.dataframe(view[["EMPLOYEE_NUMBER", "job_group", "branch_code",
                               "performance_score", "band"]].round(2),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No data matches current filters.")

# else: no file uploaded, no error — landing page already showed uploaders above.


# ── composer (pinned to bottom) ─────────────────────────────────────────
# (Removed — replaced by tile dashboard above.)