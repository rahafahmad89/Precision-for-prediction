# ===== TEST-ONLY: LOAD SAVED MODEL & PREPROCESSOR, PREDICT ON TEST CSV =====
import os, json, numpy as np, pandas as pd
import joblib, cloudpickle

# --------- PATHS (edit if needed) ----------
RESULTS_DIR  = "/content/drive/MyDrive/conv/Results_NEW24sep"
MODEL_DIR    = os.path.join(RESULTS_DIR, "model_artifacts")
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.joblib")
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
META_PATH    = os.path.join(MODEL_DIR, "meta.json")
TEST_PATH    = "/content/drive/MyDrive/conv/Test.csv"

OUT_XLSX = os.path.join(RESULTS_DIR, "external_predictions_no_labels.xlsx")

# --------- TRIAGE CUTS (high-confidence) ----------
T_LOW  = 0.2   # <= T_LOW => confident Benign
T_HIGH = 0.20   # >= T_HIGH => confident Pathogenic

# --------- Columns consistent with training ----------
ID_COLS = ['#Uploaded_variation','clinvar_hgvs','Existing_variation','Location',
           'HGVSc','HGVSp','Feature','Feature_type']
DROP_COLS_EXTRA = ['cDNA_position','CDS_position','Protein_position','AA',
                   'MANE','MANE_SELECT','MANE_PLUS_CLINICAL','FLAGS']

# --------- Helpers ----------
class ToNumericTransformer:  # for joblib compatibility if needed
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
        return X

def smart_load(path):
    try:
        with open(path, "rb") as f: return cloudpickle.load(f)
    except Exception:
        return joblib.load(path)

def get_feat_order_from_pre(pre):
    feat_order = []
    for _, _, sel in getattr(pre, "transformers_", []):
        if sel is None: continue
        try: feat_order += list(sel)
        except Exception: pass
    return feat_order

def transform_with_pre(pre, X_df, feat_order):
    for c in feat_order:
        if c not in X_df.columns: X_df[c] = np.nan
    Xt = pre.transform(X_df[feat_order].copy())
    return pd.DataFrame(Xt, columns=feat_order, index=X_df.index)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load artifacts
    meta  = json.load(open(META_PATH))
    pre   = smart_load(PREPROC_PATH)
    model = smart_load(MODEL_PATH)

    # Read test
    df = pd.read_csv(TEST_PATH, low_memory=False)

    # Choose an ID to show
    id_col = next((c for c in ['#Uploaded_variation','clinvar_hgvs','Existing_variation','Location'] if c in df.columns), None)
    ids = df[id_col] if id_col else pd.Series(np.arange(len(df)), name="ID")

    # Build features like training
    drop = [c for c in (set(ID_COLS) | set(DROP_COLS_EXTRA)) if c in df.columns]
    X_raw = df.drop(columns=drop, errors='ignore')

    feat_order = get_feat_order_from_pre(pre)
    Xt = transform_with_pre(pre, X_raw, feat_order).fillna(0.0)

    sel_feats = meta["selected_features"]
    for c in sel_feats:
        if c not in Xt.columns: Xt[c] = 0.0
    Xt = Xt.loc[:, sel_feats]

    # Predict
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xt)[:, 1]
    else:
        dec = model.decision_function(Xt)
        prob = (dec - dec.min())/(dec.ptp()+1e-9)

    thr_saved = float(meta["threshold"]["youdenJ_mean"])

    # Labels
    pred_saved = (prob >= thr_saved).astype(int)
    pred_05    = (prob >= 0.5).astype(int)

    # High-confidence triage
    triage = np.where(prob >= T_HIGH, "High-Conf Pathogenic",
             np.where(prob <= T_LOW,  "High-Conf Benign", "Gray"))

    # Save output
    out = pd.DataFrame({
        "ID": ids.values,
        "Pred_Prob": prob,
        "Pred_Label_saved_thr": pred_saved,
        "Pred_Label_0p5": pred_05,
        "Triage": triage
    })
    out.to_excel(OUT_XLSX, index=False)
    print(f"[saved] {OUT_XLSX} ({len(out)} rows)")

    # Quick counts
    print("Saved-threshold label counts:", out["Pred_Label_saved_thr"].value_counts().to_dict())
    print("Triage counts:", out["Triage"].value_counts().to_dict())
    print(f"Thresholds -> saved: {thr_saved:.3f}, triage: <= {T_LOW:.2f} / >= {T_HIGH:.2f}")

if __name__ == "__main__":
    main()
