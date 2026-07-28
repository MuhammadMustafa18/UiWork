# UBL Workforce Intelligence — UI Platform (Streamlit)

Interactive UI over the full validated pipeline (same logic as the Colab notebook v2).

## Run locally / on an internal server
    pip install -r requirements.txt
    streamlit run app.py
Then open http://localhost:8501 (add `--server.port 80 --server.address 0.0.0.0` for a shared internal server).

## Use
1. Upload the employee extract (.xlsx/.csv) in the sidebar — e.g. Latest_data.xlsx.
2. Optionally upload the branch hierarchy mapping (columns: branch_code, cluster, region[, branch_segment]).
   Without it, hierarchy is derived (region=Province, cluster=Province/City, segment=BRANCH_CATG) and flagged.
3. Adjust parameters in the sidebar (pillar weight, HP/LP thresholds, attrition/growth/promotion rates) —
   everything recomputes live; results are cached per file+parameter combination.

## Tabs
Discovery · Scores & SHAP (per-employee waterfalls) · Deciles & Hierarchy (business-unit decile
profiles, rollups, drill-downs) · HP Personas (PCA + cross-city twin finder) · LP Segments
(IF-THEN rules + diagnoses) · Growth Drivers (gender×age diagnostics, non-circular driver model)
· Hiring Recs (per branch/cluster/region/segment) · Fairness (PASS/FAIL gates) · Downloads (CSVs).

## Governance (enforced in code)
- Protected attributes (gender, age, marital status) are never model features or hiring criteria —
  diagnostics and fairness audit only.
- PII columns are dropped on load; outputs show employee number + scores only.
- Absent pillars/traits are excluded and named, never simulated. Failures (e.g. fairness FAIL)
  are displayed prominently, never softened.

## Files
- app.py — Streamlit UI
- pipeline.py — all computations (mirrors the validated notebook)
