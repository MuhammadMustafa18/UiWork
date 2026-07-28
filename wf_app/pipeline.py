"""
UBL Workforce Intelligence — core pipeline (UI backend).
All computations mirror the validated Colab notebook (v2). No plotting here.
Governance: protected attributes (GENDER, AGE, MARITAL_STATUS, ...) are never model
features; they appear only in fairness audits and diagnostic breakdowns.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBRegressor
import shap

RNG = 42
PROTECTED = {"GENDER", "AGE", "MARITAL_STATUS", "date_of_birth"}
ID_COL, DATE_COL, EFF_COL = "Employee", "Trxn_Date", "EFFECTIVE_START_DATE"

PII_DROP = ["FULL_NAME", "national_identifier", "PHONE_NUMBER", "email_address",
            "ADDRESS_LINE1", "ADDRESS_LINE2", "POSTAL_CODE", "person_title",
            "Tehsil", "Address_District", "Address_Region", "crc"]

GRADE_ORDER = {g: i for i, g in enumerate(
    ["CLR", "CON", "OG-4", "OG-3", "OG-2", "OG-1", "SC-4", "SC-3", "SC-2", "SC-1"])}

KPI_DIRECTION = {
    "trxn_count_per_day":      (+1, "More transactions per active day"),
    "checker_count_per_day":   (+1, "More verifications/checks (supervision workload)"),
    "trxn_amount_m_per_day":   (+1, "Higher value processed per day (PKR m)"),
    "total_trxn_mins_per_day": (+1, "More productive minutes on transactions"),
    "Emp_Br_benchmark":        (+1, "Higher throughput vs branch benchmark — REVIEW"),
    "Emp_Hours_Percentile":    (+1, "Higher hours percentile within branch"),
    "pct_days_above_std":      (+1, "More days rated Above Standard"),
    "pct_days_below_std":      (-1, "Fewer days rated Below Standard"),
    "pct_days_underworked":    (-1, "Fewer underworked (idle-capacity) days"),
    "active_days_ratio":       (+1, "Active on more business days (Behavioral proxy)"),
}

BANDS = [(80, 101, "High Performer"), (60, 80, "Solid"), (40, 60, "Average"),
         (20, 40, "Below Average"), (0, 20, "Low Performer")]


def _band(s):
    return next(name for a, b, name in BANDS if a <= s < b)


def _z_within(df, col, group):
    g = df.groupby(group)[col]
    mu, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).fillna(0.0).clip(-3.5, 3.5)


# ────────────────────────────────────────────────────────────────────────
def load_and_prepare(path_or_buf, hier_df=None):
    """Step 0: load, grain detection, point-in-time dedup, PII drop, aggregation."""
    review, skipped = [], []
    raw = pd.read_csv(path_or_buf) if str(getattr(path_or_buf, "name", path_or_buf)) \
        .lower().endswith(".csv") else pd.read_excel(path_or_buf)

    missing = [c for c in [ID_COL, DATE_COL] if c not in raw.columns]
    if missing:
        raise KeyError(
            f"Required column(s) missing: {missing}. "
            f"Expected employee identifier '{ID_COL}' and transaction date '{DATE_COL}'.\n"
            f"Available columns: {list(raw.columns)}")

    meta = {"rows": len(raw), "cols": raw.shape[1]}
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    meta["n_emp"] = raw[ID_COL].nunique()
    meta["n_emp_day"] = raw.groupby([ID_COL, DATE_COL]).ngroups
    meta["dup_factor"] = meta["rows"] / max(meta["n_emp_day"], 1)
    meta["window"] = (raw[DATE_COL].min(), raw[DATE_COL].max())
    meta["missing_pct"] = (raw.isna().mean() * 100).round(2).sort_values(ascending=False)

    # point-in-time dedup
    if EFF_COL in raw.columns and meta["dup_factor"] > 1.05:
        raw[EFF_COL] = pd.to_datetime(raw[EFF_COL], errors="coerce")
        r = raw.sort_values([ID_COL, DATE_COL, EFF_COL])
        valid = r[r[EFF_COL] <= r[DATE_COL]].groupby([ID_COL, DATE_COL], as_index=False).tail(1)
        have = set(map(tuple, valid[[ID_COL, DATE_COL]].values))
        fb = r[~r.set_index([ID_COL, DATE_COL]).index.isin(have)] \
            .groupby([ID_COL, DATE_COL], as_index=False).head(1)
        daily = pd.concat([valid, fb], ignore_index=True)
    else:
        daily = raw.drop_duplicates([ID_COL, DATE_COL]).copy()
    meta["dedup_removed"] = len(raw) - len(daily)

    daily = daily.drop(columns=[c for c in PII_DROP if c in daily.columns])
    network_days = daily[DATE_COL].dt.normalize().nunique()
    meta["network_days"] = network_days
    if network_days < 60:
        review.append(f"KPI window is only {network_days} business days — scores reflect a "
                      "short observation window; use a longer extract before consequential decisions.")

    # aggregation to employee grain
    daily = daily.sort_values([ID_COL, DATE_COL] + ([EFF_COL] if EFF_COL in daily.columns else []))
    num_daily = {k: v for k, v in {
        "Emp_Trxn_Count": "sum", "Emp_Checker_Count": "sum", "Emp_Trxn_Amount": "sum",
        "Emp_Total_Trxn_mins": "sum", "Emp_Br_benchmark": "mean",
        "Emp_Hours_Percentile": "mean"}.items() if k in daily.columns}
    agg = daily.groupby(ID_COL).agg(num_daily)
    agg["active_days"] = daily.groupby(ID_COL)[DATE_COL].nunique()
    for col, pref, name in [("Emp_Contribution_Remarks", "Above Standard", "pct_days_above_std"),
                            ("Emp_Contribution_Remarks", "Below Standard", "pct_days_below_std"),
                            ("Emp_Absolute_Remark", "Underworked", "pct_days_underworked"),
                            ("Emp_Absolute_Remark", "Overworked", "pct_days_overworked")]:
        if col in daily.columns:
            agg[name] = daily.groupby(ID_COL)[col].apply(lambda s, p=pref: (s == p).mean())
    agg["active_days_ratio"] = agg["active_days"] / network_days
    for base in ["Emp_Trxn_Count", "Emp_Checker_Count", "Emp_Total_Trxn_mins"]:
        if base in agg:
            agg[base.replace("Emp_", "").lower() + "_per_day"] = agg[base] / agg["active_days"]
    if "Emp_Trxn_Amount" in agg:
        agg["trxn_amount_m_per_day"] = agg["Emp_Trxn_Amount"] / agg["active_days"] / 1e6
        agg["avg_ticket_m"] = (agg["Emp_Trxn_Amount"] / agg["Emp_Trxn_Count"].replace(0, np.nan) / 1e6).fillna(0)

    attr_cols = [c for c in ["EMPLOYEE_NUMBER", "JOB", "Grade", "cadre", "sub_group", "DEPT",
                             "branch_code", "BRANCH_CATG", "City", "Province", "supervisor",
                             "position_type", "GENDER", "AGE", "YEARS_OF_SERVICE", "HIRE_DATE",
                             "stint_no", "USER_STATUS", "LEAVING_REASON", "Last_Working_date"]
                 if c in daily.columns]
    emp = daily.groupby(ID_COL).tail(1).set_index(ID_COL)[attr_cols].join(agg)

    emp["branch_headcount"] = emp.groupby("branch_code")["EMPLOYEE_NUMBER"].transform("count")
    if "supervisor" in emp:
        emp["team_size"] = emp.groupby("supervisor")["EMPLOYEE_NUMBER"].transform("count")
    emp["branch_mean_benchmark"] = emp.groupby("branch_code")["Emp_Br_benchmark"].transform("mean")
    emp["grade_level"] = emp["Grade"].map(GRADE_ORDER)
    review.append("'team_size'/'branch_headcount' count colleagues within this extract only — "
                  "lower bounds on true spans.")
    review.append("Grade seniority assumed OG-1 > … > OG-4, SC above OG — confirm the ladder.")

    core = [c for c in ["trxn_count_per_day", "checker_count_per_day", "trxn_amount_m_per_day",
                        "total_trxn_mins_per_day", "Emp_Br_benchmark", "Emp_Hours_Percentile",
                        "pct_days_above_std", "pct_days_below_std", "pct_days_underworked",
                        "active_days_ratio"] if c in emp.columns]
    emp["low_confidence"] = emp[core].isna().mean(axis=1) > 0.5

    impute_log = []
    for c in emp.columns:
        n = emp[c].isna().sum()
        if n == 0 or c in ("HIRE_DATE", "LEAVING_REASON", "Last_Working_date"):
            continue
        if pd.api.types.is_numeric_dtype(emp[c]):
            emp[c] = emp[c].fillna(emp[c].median()); how = "median"
        else:
            m = emp[c].mode(); emp[c] = emp[c].fillna(m.iloc[0] if len(m) else "UNKNOWN"); how = "mode"
        impute_log.append((c, int(n), how))
    meta["impute_log"] = pd.DataFrame(impute_log, columns=["column", "n_imputed", "method"])

    # pillar availability (never simulated)
    meta["pillars"] = pd.DataFrame([
        ("Sales & Revenue", "NONE", "excluded"), ("Customer Experience", "NONE", "excluded"),
        ("Operational Efficiency", "transaction throughput columns", "USED (80%)"),
        ("Compliance & Risk", "NONE", "excluded"),
        ("Behavioral", "active-days proxy (derived)", "USED (20%) — proxy, REVIEW")],
        columns=["Pillar", "Usable data", "Status"])
    review.append("Only Operational Efficiency has real KPIs; the score is a THROUGHPUT score, "
                  "not holistic performance, until sales/CX/compliance data is joined.")
    review.append("Behavioral pillar = active-transaction-days proxy; Emp_Br_benchmark direction "
                  "assumed higher-is-better — confirm with HR/MIS.")

    n_leavers = int(emp["LEAVING_REASON"].notna().sum()) if "LEAVING_REASON" in emp else 0
    if n_leavers < 30:
        skipped.append(f"Attrition model (only {n_leavers} exit signals — too few to train).")
    if not any(c in emp.columns for c in ("EDUCATION", "QUALIFICATION")):
        skipped.append("Education-based profiling (no education column).")

    # hierarchy
    hier_source = "DERIVED — region=Province, cluster=Province/City, segment=BRANCH_CATG"
    if hier_df is not None:
        h = hier_df.copy(); h.columns = [c.strip().lower() for c in h.columns]
        if {"branch_code", "cluster", "region"}.issubset(h.columns):
            keep = ["branch_code", "cluster", "region"] + \
                   (["branch_segment"] if "branch_segment" in h.columns else [])
            emp = emp.reset_index().merge(h[keep].drop_duplicates("branch_code"),
                                          on="branch_code", how="left").set_index(ID_COL)
            hier_source = "uploaded branch-hierarchy mapping file"
    if "region" not in emp.columns or emp.get("region", pd.Series(dtype=object)).isna().any():
        emp["region"] = emp.get("region", pd.Series(index=emp.index, dtype=object))
        emp["region"] = emp["region"].fillna(emp["Province"])
    if "cluster" not in emp.columns or emp["cluster"].isna().any():
        emp["cluster"] = emp.get("cluster", pd.Series(index=emp.index, dtype=object))
        emp["cluster"] = emp["cluster"].fillna(emp["Province"].astype(str) + " / " + emp["City"].astype(str))
    if "branch_segment" not in emp.columns:
        emp["branch_segment"] = emp["BRANCH_CATG"]
    if hier_source.startswith("DERIVED"):
        review.append("No branch→cluster→region mapping uploaded — hierarchy derived from "
                      "address geography + BRANCH_CATG. Upload the org mapping for true rollups.")
    meta["hier_source"] = hier_source
    return emp, meta, review, skipped


# ────────────────────────────────────────────────────────────────────────
def score_employees(emp, w_ops=0.8, min_job_n=8):
    emp = emp.copy()
    kdir = {k: v for k, v in KPI_DIRECTION.items() if k in emp.columns}
    sizes = emp["JOB"].value_counts()
    emp["job_group"] = emp["JOB"].where(emp["JOB"].map(sizes) >= min_job_n, "Pooled (small roles)")
    for k, (d, _) in kdir.items():
        emp[f"z__{k}"] = d * _z_within(emp, k, "job_group")
    ops = [k for k in kdir if k != "active_days_ratio"]
    emp["pillar_ops_z"] = emp[[f"z__{k}" for k in ops]].mean(axis=1)
    emp["pillar_beh_z"] = emp["z__active_days_ratio"] if "z__active_days_ratio" in emp else 0.0
    emp["ps_raw"] = w_ops * emp["pillar_ops_z"] + (1 - w_ops) * emp["pillar_beh_z"]
    lo, hi = emp["ps_raw"].min(), emp["ps_raw"].max()
    emp["performance_score"] = 1 + 99 * (emp["ps_raw"] - lo) / (hi - lo)
    emp["band"] = emp["performance_score"].apply(_band)

    def dec(g):
        if len(g) < 20:
            return pd.Series("n/a", index=g.index)
        r = g["performance_score"].rank(pct=True, method="first")
        return "D" + np.ceil((1 - r) * 10).clip(1, 10).astype(int).astype(str)
    emp["decile"] = emp.groupby("job_group", group_keys=False).apply(dec)
    emp["is_top_decile"] = emp["decile"] == "D1"

    bsum = emp.groupby("branch_code")["performance_score"].transform("sum")
    bcnt = emp.groupby("branch_code")["performance_score"].transform("count")
    emp["branch_health"] = np.where(bcnt > 1, (bsum - emp["performance_score"]) / (bcnt - 1),
                                    emp["performance_score"].mean())
    dir_tbl = pd.DataFrame([(k, "higher = better" if d > 0 else "LOWER = better", why)
                            for k, (d, why) in kdir.items()],
                           columns=["KPI", "direction", "rationale"])
    return emp, dir_tbl


def surrogate_model(emp):
    feats = [c for c in ["trxn_count_per_day", "checker_count_per_day", "trxn_amount_m_per_day",
                         "total_trxn_mins_per_day", "Emp_Br_benchmark", "Emp_Hours_Percentile",
                         "pct_days_above_std", "pct_days_below_std", "pct_days_underworked",
                         "pct_days_overworked", "active_days_ratio", "active_days",
                         "YEARS_OF_SERVICE", "grade_level", "branch_headcount", "team_size",
                         "branch_mean_benchmark", "stint_no"]
             if c in emp.columns and c not in PROTECTED]
    X, y = emp[feats].astype(float), emp["performance_score"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RNG)
    m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9, random_state=RNG).fit(Xtr, ytr)
    r2 = r2_score(yte, m.predict(Xte))
    rmse = float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    sv = shap.TreeExplainer(m)(X)
    model_score = pd.Series(m.predict(X), index=emp.index)
    return dict(model=m, features=feats, X=X, shap=sv, r2=r2, rmse=rmse, model_score=model_score)


def driver_text(sv, feats, i, top=5):
    row = pd.Series(sv.values[i], index=feats)
    t = row.reindex(row.abs().sort_values(ascending=False).head(top).index)
    return "; ".join(f"{f} {'raises' if v > 0 else 'lowers'} score by {abs(v):.1f}" for f, v in t.items())


# ────────────────────────────────────────────────────────────────────────
def _choose_k(X, lo=2, hi=8):
    ks = list(range(lo, max(lo + 1, hi + 1)))
    sil = []
    for k in ks:
        if k >= len(X):
            break
        sil.append((k, silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=RNG).fit_predict(X))))
    if not sil:
        return 2, []
    best = max(sil, key=lambda t: t[1])[0]
    return best, sil


def hp_engine(emp, threshold=80):
    hp = emp[emp["performance_score"] >= threshold].copy()
    rule = f"score ≥ {threshold}"
    if len(hp) < 15:
        thr = float(np.percentile(emp["performance_score"], 90))
        hp = emp[emp["performance_score"] >= thr].copy()
        rule = f"top decile (score ≥ {thr:.1f}) — fallback, <15 above {threshold}"
    feats = [c for c in ["pillar_ops_z", "pillar_beh_z", "YEARS_OF_SERVICE", "active_days_ratio",
                         "trxn_amount_m_per_day", "trxn_count_per_day", "checker_count_per_day",
                         "Emp_Hours_Percentile", "team_size", "branch_headcount"] if c in hp.columns]
    X = StandardScaler().fit_transform(hp[feats].astype(float))
    k, sil = _choose_k(X, 2, min(8, max(3, len(hp) // 8)))
    hp["hp_cluster"] = KMeans(n_clusters=k, n_init=25, random_state=RNG).fit_predict(X)
    P = PCA(n_components=2, random_state=RNG).fit_transform(X)
    prof = hp.groupby("hp_cluster")[feats].mean().round(2)
    LBL = {"pillar_ops_z": "Throughput", "pillar_beh_z": "Consistency", "YEARS_OF_SERVICE": "Tenure",
           "active_days_ratio": "Presence", "trxn_amount_m_per_day": "Value-Volume",
           "trxn_count_per_day": "Count-Volume", "checker_count_per_day": "Checker-Load",
           "Emp_Hours_Percentile": "Long-Hours", "team_size": "Big-Team", "branch_headcount": "Big-Branch"}
    cz = (prof - hp[feats].mean()) / hp[feats].std().replace(0, np.nan)
    names = {}
    for c in prof.index:
        dev = cz.loc[c].dropna().sort_values()
        hi_ = [LBL.get(f, f) for f in dev.tail(2).index[::-1] if dev[f] > 0.25]
        lo_ = [f"Low-{LBL.get(f, f)}" for f in dev.head(1).index if dev[f] < -0.25]
        names[c] = " · ".join((hi_ + lo_) or ["Balanced All-Rounder"]) + " HP"
    hp["persona"] = hp["hp_cluster"].map(names)
    S = cosine_similarity(X); np.fill_diagonal(S, -1)
    return dict(hp=hp, rule=rule, feats=feats, k=k, sil=sil, pca=P, profile=prof,
                personas=names, sim=S)


def similar_in_other_cities(hp_res, emp_number, n=5, city_col="City"):
    hp = hp_res["hp"]
    idx = hp.index[hp["EMPLOYEE_NUMBER"] == emp_number]
    if len(idx) == 0:
        return None, None
    i = list(hp.index).index(idx[0])
    other = (hp[city_col].values != hp[city_col].values[i])
    sims = pd.Series(hp_res["sim"][i], index=hp.index)[other].sort_values(ascending=False).head(n)
    out = hp.loc[sims.index, ["EMPLOYEE_NUMBER", "job_group", city_col, "region", "persona",
                              "performance_score"]].copy()
    out["similarity_%"] = (sims.values * 100).round(1)
    return hp.loc[idx[0]], out


def lp_engine(emp, threshold=40):
    lp = emp[emp["performance_score"] < threshold].copy()
    rule = f"score < {threshold}"
    if len(lp) < 15:
        thr = float(np.percentile(emp["performance_score"], 10))
        lp = emp[emp["performance_score"] <= thr].copy()
        rule = f"bottom decile — fallback, <15 under {threshold}"
    feats = [c for c in ["YEARS_OF_SERVICE", "grade_level", "active_days_ratio",
                         "pct_days_underworked", "pct_days_below_std", "branch_health",
                         "branch_headcount", "team_size", "branch_mean_benchmark"] if c in lp.columns]
    X = StandardScaler().fit_transform(lp[feats].astype(float))
    k, sil = _choose_k(X, 2, min(6, max(3, len(lp) // 10)))
    lp["lp_segment"] = KMeans(n_clusters=k, n_init=25, random_state=RNG).fit_predict(X)
    prof = lp.groupby("lp_segment")[feats].mean().round(2)
    prof["n"] = lp["lp_segment"].value_counts().sort_index()
    dt = DecisionTreeClassifier(max_depth=3, random_state=RNG, class_weight="balanced")
    dt.fit(lp[feats].astype(float), lp["lp_segment"])
    rules = export_text(dt, feature_names=feats)

    wf_mean, wf_std = emp[feats].mean(), emp[feats].std().replace(0, np.nan)
    diags = []
    for seg in sorted(lp["lp_segment"].unique()):
        g = lp[lp["lp_segment"] == seg]
        zz = ((g[feats].mean() - wf_mean) / wf_std).fillna(0)
        tenure = g["YEARS_OF_SERVICE"].mean()
        if zz.get("branch_health", 0) < -0.40:
            cause, action = ("Environmental — branches whose other staff also score low "
                             f"(branch health {g['branch_health'].mean():.1f} vs workforce "
                             f"{emp['branch_health'].mean():.1f})",
                             "Branch/manager intervention first; do not PIP in a failing environment")
        elif tenure < 1.5:
            cause, action = (f"Training / onboarding gap — avg tenure only {tenure:.1f}y",
                             "Structured L&D + buddy program; re-score after 2 quarters")
        elif zz.get("active_days_ratio", 0) < -0.40 or zz.get("pct_days_underworked", 0) > 0.40:
            cause, action = (f"Engagement deficit — active {g['active_days_ratio'].mean():.0%} of days, "
                             f"underworked {g['pct_days_underworked'].mean():.0%}",
                             "Manager 1:1s, workload redesign, stay interviews")
        elif tenure > emp["YEARS_OF_SERVICE"].median() * 1.5:
            cause, action = (f"Persistent underperformance — tenure {tenure:.1f}y well above median",
                             "Formal PIP with documented targets + role-fit assessment")
        else:
            cause, action = ("Role-fit mismatch — normal presence, healthy branches",
                             "Redeployment assessment against HP personas")
        diags.append(dict(segment=seg, n=len(g), tenure=round(tenure, 1),
                          active=f"{g['active_days_ratio'].mean():.0%}",
                          branch_health=round(g["branch_health"].mean(), 1),
                          diagnosis=cause, action=action))
    return dict(lp=lp, rule=rule, feats=feats, k=k, profile=prof, rules=rules,
                diagnoses=pd.DataFrame(diags))


# ────────────────────────────────────────────────────────────────────────
def growth_engine(emp):
    comp = [c for c in ["trxn_amount_m_per_day", "trxn_count_per_day", "Emp_Br_benchmark"]
            if f"z__{c}" in emp.columns]
    emp = emp.copy()
    emp["growth_index"] = emp[[f"z__{c}" for c in comp]].mean(axis=1)
    thr = float(emp["growth_index"].quantile(0.90))
    emp["growth_driver"] = emp["growth_index"] >= thr
    emp["age_band"] = pd.cut(emp["AGE"], [18, 28, 35, 42, 200],
                             labels=["21–27", "28–34", "35–41", "42+"], right=False)
    cell = emp.groupby(["GENDER", "age_band"], observed=True).agg(
        headcount=("EMPLOYEE_NUMBER", "count"),
        avg_amount_m_day=("trxn_amount_m_per_day", "mean"),
        avg_growth_index=("growth_index", "mean"),
        growth_driver_rate=("growth_driver", "mean")).round(3)

    for c, src in [("branch_categ_code", "BRANCH_CATG"), ("cadre_code", "cadre"),
                   ("sub_group_code", "sub_group"), ("position_type_code", "position_type")]:
        if src in emp.columns:
            emp[c] = emp[src].astype("category").cat.codes
    dfeats = [c for c in ["YEARS_OF_SERVICE", "grade_level", "active_days_ratio",
                          "branch_headcount", "team_size", "branch_categ_code", "cadre_code",
                          "sub_group_code", "position_type_code", "stint_no"] if c in emp.columns]
    Xd, yd = emp[dfeats].astype(float), emp["trxn_amount_m_per_day"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=RNG)
    m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9, random_state=RNG).fit(Xtr, ytr)
    r2 = r2_score(yte, m.predict(Xte))
    sv = shap.TreeExplainer(m)(Xd)
    imp = pd.Series(np.abs(sv.values).mean(0), index=dfeats).sort_values(ascending=False)

    lifts = []
    for f in imp.head(6).index:
        j = dfeats.index(f)
        corr = np.corrcoef(Xd[f], sv.values[:, j])[0, 1] if Xd[f].std() > 0 else 0.0
        d = "higher is better" if corr > 0.05 else ("lower is better" if corr < -0.05 else "mixed")
        q1, q3 = Xd[f].quantile(0.25), Xd[f].quantile(0.75)
        lo_m, hi_m = yd[Xd[f] <= q1].mean(), yd[Xd[f] >= q3].mean()
        lifts.append((f, d, round(hi_m, 2), round(lo_m, 2),
                      round((hi_m - lo_m) / lo_m * 100, 0) if lo_m else np.nan))
    lifts = pd.DataFrame(lifts, columns=["feature", "direction", "top_quartile_mean",
                                         "bottom_quartile_mean", "lift_%"])
    imp_by_g = {}
    for g in emp["GENDER"].dropna().unique():
        msk = (emp["GENDER"] == g).values
        if msk.sum() >= 30:
            s = pd.Series(np.abs(sv.values[msk]).mean(0), index=dfeats)
            imp_by_g[g] = (s / s.sum()).round(3)
    wf_mix = emp["GENDER"].value_counts(normalize=True)
    gd_mix = emp.loc[emp["growth_driver"], "GENDER"].value_counts(normalize=True) \
        .reindex(wf_mix.index).fillna(0)
    parity_gap_pp = float((gd_mix - wf_mix).abs().max() * 100)
    return dict(emp=emp, cell=cell, dfeats=dfeats, X=Xd, shap=sv, r2=r2, imp=imp,
                lifts=lifts, imp_by_gender=pd.DataFrame(imp_by_g), parity_gap_pp=parity_gap_pp)


# ────────────────────────────────────────────────────────────────────────
TRAIT_CANDIDATES = {
    "tenure_years": ["YEARS_OF_SERVICE"], "grade": ["grade_level"],
    "experience_stints": ["stint_no"], "education": ["EDUCATION", "QUALIFICATION"],
    "discipline_record": ["DISCIPLINE"], "audit_rating": ["AUDIT_RATING", "audit_score"],
    "salary": ["SALARY", "BASIC_SALARY", "gross_salary"],
    "presence": ["active_days_ratio"], "hours_intensity": ["Emp_Hours_Percentile"],
    "team_size": ["team_size"], "branch_size": ["branch_headcount"],
}
HI_LBL = {"tenure_years": "Veteran", "grade": "Senior-Grade", "experience_stints": "Multi-Stint",
          "presence": "Ever-Present", "hours_intensity": "Long-Hours", "team_size": "Team-Embedded",
          "branch_size": "Big-Branch", "education": "Well-Qualified", "audit_rating": "Clean-Audit",
          "salary": "Higher-Paid", "productivity_per_salary": "High-ROI"}
LO_LBL = {"tenure_years": "Early-Career", "grade": "Junior-Grade", "presence": "Low-Presence",
          "hours_intensity": "Short-Hours", "team_size": "Solo-Operator",
          "branch_size": "Small-Branch", "productivity_per_salary": "Low-ROI"}


def trait_personas(emp, top_deciles=("D1", "D2", "D3"), geo=None):
    traits, missing = {}, []
    for t, cands in TRAIT_CANDIDATES.items():
        col = next((c for c in cands if c in emp.columns), None)
        traits.__setitem__(t, col) if col else missing.append(t)
    pool = emp[emp["decile"].isin(top_deciles)].copy()
    if geo:
        pool = pool[pool[geo[0]] == geo[1]]
    tcols = list(traits.values())
    if traits.get("salary"):
        pool["productivity_per_salary"] = pool["trxn_amount_m_per_day"] / pool[traits["salary"]].replace(0, np.nan)
        tcols = tcols + ["productivity_per_salary"]
    results = {}
    col2trait = {v: k for k, v in traits.items()}
    col2trait["productivity_per_salary"] = "productivity_per_salary"
    pool["trait_persona"] = pd.Series(pd.NA, index=pool.index, dtype="object")
    for job, g in pool.groupby("job_group"):
        if len(g) < 24:
            fp = g[tcols].mean().round(2)
            results[job] = dict(mode="fingerprint", n=len(g), fingerprint=fp,
                                diag=pd.DataFrame({"pct_female": [(g["GENDER"] == "F").mean()],
                                                   "mean_age": [g["AGE"].mean()]}).round(2))
            continue
        X = StandardScaler().fit_transform(g[tcols].astype(float).fillna(g[tcols].mean()))
        k, _ = _choose_k(X, 2, min(6, len(g) // 10))
        while k >= 2:
            lab = KMeans(n_clusters=k, n_init=25, random_state=RNG).fit_predict(X)
            if pd.Series(lab).value_counts().min() >= 3:
                break
            k -= 1
        g = g.copy(); g["tc"] = lab if k >= 2 else 0
        prof = g.groupby("tc")[tcols].mean().round(2)
        prof["n"] = g["tc"].value_counts().sort_index()
        cz = (prof[tcols] - g[tcols].mean()) / g[tcols].std().replace(0, np.nan)
        names = {}
        for c in prof.index:
            dev = cz.loc[c].dropna().sort_values()
            hi_ = [HI_LBL.get(col2trait.get(t, t), t.title()) for t in dev.tail(2).index[::-1] if dev[t] > 0.3]
            lo_ = [LO_LBL.get(col2trait.get(t, t), f"Low-{t.title()}") for t in dev.head(1).index if dev[t] < -0.3]
            names[c] = " ".join(dict.fromkeys(hi_ + lo_)) or "Balanced Professional"
        g["trait_persona"] = g["tc"].map(names)
        pool.loc[g.index, "trait_persona"] = g["trait_persona"]
        prof.index = [f"{names[c]}" for c in prof.index]
        diag = g.groupby("trait_persona").agg(pct_female=("GENDER", lambda s: (s == "F").mean()),
                                              mean_age=("AGE", "mean")).round(2)
        results[job] = dict(mode="clusters", n=len(g), k=k, profile=prof, diag=diag)
    return pool, results, traits, missing


def rollup(emp, level, min_n=8):
    g = emp[emp["decile"].str.startswith("D", na=False)].groupby([level, "job_group"], observed=True)
    t = g.agg(n=("EMPLOYEE_NUMBER", "count"), median_score=("performance_score", "median"),
              top_decile_pct=("is_top_decile", "mean"),
              hp_band_pct=("band", lambda s: (s == "High Performer").mean()),
              txns_day=("trxn_count_per_day", "mean"), avg_ticket_m=("avg_ticket_m", "mean"),
              value_day_m=("trxn_amount_m_per_day", "mean")).round(2)
    t["top_decile_pct"] = (t["top_decile_pct"] * 100).round(1)
    t["hp_band_pct"] = (t["hp_band_pct"] * 100).round(1)
    t["stable"] = np.where(t["n"] >= min_n, "yes", "no (n<%d)" % min_n)
    return t.reset_index()


def hiring_recs(emp, pool, level, attr_rate, growth_rate, promo_rate, min_n=8):
    rows = []
    for node, g in emp.groupby(level, observed=True):
        hc = len(g)
        need = max(0, round(hc * (attr_rate + growth_rate - promo_rate)))
        dom = g["job_group"].mode().iloc[0]
        jp = pool[pool["job_group"] == dom]
        persona, fp_line = "Balanced Professional", "—"
        if "trait_persona" in jp.columns and jp["trait_persona"].notna().any():
            q = jp.groupby("trait_persona")["performance_score"].mean()
            elig = q[q >= q.median()].index
            net = jp.loc[jp["trait_persona"].isin(elig), "trait_persona"].value_counts(normalize=True)
            node_share = g.loc[g["job_group"] == dom].index
            ns = jp.loc[jp.index.isin(node_share), "trait_persona"].value_counts(normalize=True)
            gap = net - ns.reindex(net.index).fillna(0)
            if len(gap):
                persona = gap.idxmax()
                fp = jp[jp["trait_persona"] == persona]
                if len(fp) >= 3:
                    fp_line = (f"tenure {fp['YEARS_OF_SERVICE'].quantile(.25):.1f}–"
                               f"{fp['YEARS_OF_SERVICE'].quantile(.75):.1f}y, grade "
                               f"{fp['Grade'].mode().iloc[0]}, active {fp['active_days_ratio'].mean():.0%}, "
                               f"~{fp['trxn_count_per_day'].mean():.0f} txns/day @ PKR "
                               f"{fp['avg_ticket_m'].mean():.1f}m ticket")
        rows.append((node, hc, round(hc * attr_rate, 1), round(hc * growth_rate, 1),
                     round(hc * promo_rate, 1), need, dom, persona, fp_line,
                     "yes" if hc >= min_n else "no"))
    return pd.DataFrame(rows, columns=[level, "headcount", "projected_attrition", "growth_target",
                                       "promotion_pipeline", "net_hire_need", "dominant_job",
                                       "target_persona", "screening_fingerprint", "stable"]) \
        .sort_values("net_hire_need", ascending=False)


# ────────────────────────────────────────────────────────────────────────
def fairness_audit(emp, model_score, parity_pp=5.0, calib_pts=3.0):
    emp = emp.copy(); emp["model_score"] = model_score
    if "age_band" not in emp:
        emp["age_band"] = pd.cut(emp["AGE"], [18, 28, 35, 42, 200],
                                 labels=["21–27", "28–34", "35–41", "42+"], right=False)
    tables, checks = {}, []
    for dim in ["GENDER", "age_band"]:
        groups = emp.groupby(dim, observed=True)
        stable = groups.size()[groups.size() >= 10].index
        hp_r = groups.apply(lambda g: (g["band"] == "High Performer").mean()).loc[stable] * 100
        lp_r = groups.apply(lambda g: g["band"].isin(["Low Performer", "Below Average"]).mean()).loc[stable] * 100
        qthr = emp["ps_raw"].quantile(0.75)
        eo = groups.apply(lambda g: (g.loc[g["ps_raw"] >= qthr, "band"] == "High Performer").mean()
                          if (g["ps_raw"] >= qthr).sum() >= 5 else np.nan).loc[stable] * 100
        cal = groups.apply(lambda g: (g["model_score"] - g["performance_score"]).abs().mean()).loc[stable]
        tables[dim] = pd.DataFrame({"n": groups.size().loc[stable], "HP_rate_%": hp_r.round(1),
                                    "LP+BA_rate_%": lp_r.round(1),
                                    "equal_opp_%": eo.round(1), "calibration": cal.round(2)})
        for name, gap, thr in [("Demographic parity (HP rate)", hp_r.max() - hp_r.min(), parity_pp),
                               ("Demographic parity (LP rate)", lp_r.max() - lp_r.min(), parity_pp),
                               ("Equal opportunity", (np.nanmax(eo) - np.nanmin(eo))
                                if eo.notna().sum() >= 2 else 0.0, parity_pp),
                               ("Calibration gap", cal.max() - cal.min(), calib_pts)]:
            checks.append((dim, name, round(float(gap), 1), thr,
                           "PASS" if gap <= thr else "FAIL"))
    checks = pd.DataFrame(checks, columns=["dimension", "check", "gap", "threshold", "result"])
    return tables, checks, bool((checks["result"] == "FAIL").any())
