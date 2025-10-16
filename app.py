# ==========================================================
# app.py — Streamlit churn scoring + interpretability (XGBoost + SHAP)
# ==========================================================
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# plotting
import matplotlib.pyplot as plt

# shap (tree-based fast explainer)
import shap




@st.cache_data(show_spinner=False)
def predict_proba_cached(_model, X: pd.DataFrame) -> np.ndarray:
    """
    Cached probability predictions to avoid recomputation.
    Streamlit will skip hashing `_model` since the arg name starts with '_'.
    """
    return _model.predict_proba(X)[:, 1]

# ---- constants ----
DEFAULT_MODEL_PATH = Path("models/retention_xgb.joblib")
DEFAULT_META_PATH  = Path("models/model_meta.json")

st.set_page_config(page_title="Churn Scorer & Explainability", page_icon="📉", layout="wide")
st.title("📉 Churn Prediction — Scoring & Explainability")
st.caption("Upload a CSV of members. We’ll score churn probability, and show global + local explanations.")

# ======================
# Sidebar: model inputs
# ======================
st.sidebar.header("Model")
model_file = st.sidebar.file_uploader("Upload a .joblib model (optional)", type=["joblib"])
meta_file  = st.sidebar.file_uploader("Upload model_meta.json (optional)", type=["json"])

# ----------------------
# Load model & metadata
# ----------------------
@st.cache_resource(show_spinner=False)
def load_pipeline_and_meta(model_bytes: bytes | None, meta_bytes: bytes | None):
    # model
    if model_bytes is not None:
        model = joblib.load(io.BytesIO(model_bytes))
    else:
        if not DEFAULT_MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Upload a .joblib or place it at models/retention_xgb.joblib")
        model = joblib.load(DEFAULT_MODEL_PATH)

    # meta
    if meta_bytes is not None:
        meta = json.loads(meta_bytes.decode("utf-8"))
    else:
        if DEFAULT_META_PATH.exists():
            meta = json.load(open(DEFAULT_META_PATH))
        else:
            # minimal fallback if meta missing
            meta = {
                "features_cat": ['MembershipSubCategory','MembershipTerm','MembershipType','PaymentFrequency','Gender'],
                "features_num": ['RegularPayment','Age','TotalAttendance'],
                "target": "Churned",
                "threshold_default": 0.5
            }
    return model, meta

try:
    model, meta = load_pipeline_and_meta(
        model_file.read() if model_file else None,
        meta_file.read() if meta_file else None
    )
except Exception as e:
    st.error(f"Error loading model/meta: {e}")
    st.stop()

# schema + threshold
features_cat = meta["features_cat"]
features_num = meta["features_num"]
expected_cols = features_cat + features_num
target_col    = meta.get("target", "Churned")
default_thresh = float(meta.get("threshold_default", 0.5))

# Decision threshold
st.sidebar.header("Scoring")
threshold = st.sidebar.slider(
    "Decision threshold (positive if prob ≥ threshold)",
    min_value=0.05, max_value=0.95, value=default_thresh, step=0.01
)

# ----------------------
# Tabs
# ----------------------
tab_score, tab_importance, tab_shap_summary, tab_shap_local = st.tabs(
    ["⚙️ Score CSV", "🌍 Global Importance", "📊 SHAP Summary", "👤 Individual Explanation"]
)

# ======================
# Common helpers
# ======================
def ensure_expected_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    return df


def get_feature_names(model) -> list[str]:
    """Get transformed feature names after preprocessor."""
    preproc = model.named_steps.get("preprocessor") or model.named_steps.get("prep")
    try:
        return list(preproc.get_feature_names_out())
    except Exception:
        return [f"f_{i}" for i in range(model.named_steps["preprocessor"].transform(np.zeros((1, len(expected_cols)))).shape[1])]

# ================
# TAB 1: Scoring
# ================
with tab_score:
    uploaded = st.file_uploader("Upload members CSV", type=["csv"])
    st.caption("Expected columns include: " + ", ".join(expected_cols) + ". Extra columns are ignored; missing ones are imputed by the pipeline.")

    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.dataframe(df_in.head(10), use_container_width=True)

        # prepare features
        df_in = ensure_expected_columns(df_in, expected_cols)
        X = df_in[expected_cols].copy()

        # predict
        try:
            proba = predict_proba_cached(model, X)
            pred  = (proba >= threshold).astype(int)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        out = df_in.copy()
        out["churn_probability"] = proba
        out["churn_pred"] = pred

        # Optional: if target is present, show quick metrics
        if target_col in df_in.columns:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
                y_true = df_in[target_col].astype(int).values
                y_hat  = pred
                auc    = roc_auc_score(y_true, proba)
                acc    = accuracy_score(y_true, y_hat)
                prec   = precision_score(y_true, y_hat, zero_division=0)
                rec    = recall_score(y_true, y_hat, zero_division=0)
                f1     = f1_score(y_true, y_hat, zero_division=0)
                cm     = confusion_matrix(y_true, y_hat)

                st.markdown("### Validation (from uploaded file)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("ROC AUC", f"{auc:.3f}")
                m2.metric("Accuracy", f"{acc:.3f}")
                m3.metric("Precision", f"{prec:.3f}")
                m4.metric("Recall", f"{rec:.3f}")
                m5.metric("F1", f"{f1:.3f}")

                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha="center", va="center")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not compute validation metrics: {e}")

        # summary + download
        st.subheader("Scored Output")
        l, r = st.columns(2)
        with l:
            st.metric("Records scored", len(out))
        with r:
            st.metric(f"Predicted churn (≥ {threshold:.2f})", int(out["churn_pred"].sum()))

        st.dataframe(out.head(20), use_container_width=True)
        st.download_button(
            label="⬇️ Download scored CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="scored_members.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload a CSV to begin scoring.")

# ==========================
# TAB 2: Global Importance
# ==========================
with tab_importance:
    st.markdown("#### Which features matter most overall?")
    st.caption("Model-reported importance (XGBoost). Larger bars indicate stronger influence on predictions.")

    try:
        xgb = model.named_steps["model"]
        # If preprocessor exists, show transformed names; otherwise base features
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names = expected_cols

        importances = getattr(xgb, "feature_importances_", None)
        if importances is None:
            st.warning("This model does not expose feature_importances_.")
        else:
            fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            fi = fi.sort_values("Importance", ascending=False).head(30)

            fig, ax = plt.subplots(figsize=(8, min(10, 0.35 * len(fi))))
            ax.barh(fi["Feature"], fi["Importance"])
            ax.invert_yaxis()
            ax.set_title("Top Feature Importances")
            ax.set_xlabel("Importance")
            st.pyplot(fig)
            st.dataframe(fi.reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")

# =======================
# TAB 3: SHAP Summary
# =======================
with tab_shap_summary:
    st.markdown("#### SHAP Summary (Global Directionality)")
    st.caption("Shows how higher/lower values move churn probability across the dataset. Each dot = one member.")

    data_file = st.file_uploader("Upload a CSV to compute SHAP summary (can reuse the scoring file)", type=["csv"], key="shap_sum_csv")
    max_rows = st.slider("Sample up to N rows for SHAP (to keep it fast)", min_value=200, max_value=5000, value=1500, step=100)

    if data_file:
        df_raw = pd.read_csv(data_file)
        df_raw = ensure_expected_columns(df_raw, expected_cols)
        X_raw  = df_raw[expected_cols].copy()

        # sample to speed up SHAP
        if len(X_raw) > max_rows:
            X_raw = X_raw.sample(n=max_rows, random_state=42)

        preproc = model.named_steps["preprocessor"]
        xgb     = model.named_steps["model"]
        try:
            transformed = preproc.transform(X_raw)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

        # SHAP expects numpy array; map back names
        try:
            transformed_names = preproc.get_feature_names_out()
        except Exception:
            transformed_names = [f"f_{i}" for i in range(transformed.shape[1])]

        st.write("Computing SHAP values… (for sampled rows)")
        # Fast TreeExplainer for tree models
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(transformed)

        # Summary plot
        fig = plt.figure()
        shap.summary_plot(shap_values, features=transformed, feature_names=transformed_names, show=False)
        st.pyplot(fig, clear_figure=True)

# ============================
# TAB 4: SHAP Local (Waterfall)
# ============================
with tab_shap_local:
    st.markdown("#### Individual Explanation (Waterfall)")
    st.caption("Pick one member to see which features pushed the prediction up or down.")

    data_file_local = st.file_uploader("Upload a CSV (same schema) for local explanation", type=["csv"], key="shap_local_csv")
    idx = st.number_input("Row index to explain", min_value=0, value=0, step=1)

    if data_file_local:
        df_local = pd.read_csv(data_file_local)
        df_local = ensure_expected_columns(df_local, expected_cols)
        X_local  = df_local[expected_cols].copy()

        preproc = model.named_steps["preprocessor"]
        xgb     = model.named_steps["model"]

        try:
            transformed_local = preproc.transform(X_local)
            transformed_names = preproc.get_feature_names_out()
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

        if idx >= transformed_local.shape[0]:
            st.warning(f"Row index {idx} is out of range (0 .. {transformed_local.shape[0]-1}).")
        else:
            explainer = shap.TreeExplainer(xgb)
            # New SHAP API returns Explanation; older returns arrays. Handle both:
            try:
                shap_values_local = explainer(transformed_local[idx])
                fig = shap.plots.waterfall(shap_values_local, show=False)
                st.pyplot(bbox_inches="tight", clear_figure=True)
            except Exception:
                # fallback to classic API
                sv = explainer.shap_values(transformed_local)
                base_value = explainer.expected_value
                # If multiclass, select positive class:
                if isinstance(sv, list):
                    sv = sv[1]
                    base_value = base_value[1]
                shap.waterfall_plot(shap.Explanation(values=sv[idx],
                                                     base_values=base_value,
                                                     data=transformed_local[idx],
                                                     feature_names=transformed_names), show=False)
                st.pyplot(bbox_inches="tight", clear_figure=True)

            # also show the raw row + predicted probability
            proba = model.predict_proba(X_local.iloc[[idx]])[:, 1][0]
            pred  = int(proba >= threshold)
            st.metric("Churn probability", f"{proba:.3f}")
            st.metric("Predicted churn", "Yes" if pred == 1 else "No")
            st.subheader("Member row (original columns)")
            st.dataframe(df_local.iloc[[idx]], use_container_width=True)
