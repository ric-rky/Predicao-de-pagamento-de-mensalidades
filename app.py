import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# Optional (pode demorar/ser pesado)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


APP_TITLE = "Next Best Action — Collections (Tuition Payments)"
DATA_DIR = Path(__file__).resolve().parent / "dados"
MENSALIDADES_PATH = DATA_DIR / "mensalidades_teste.csv"
COBRANCAS_PATH = DATA_DIR / "cobrancas_teste.csv"


# ---------------------------
# Data & features
# ---------------------------
@st.cache_data(show_spinner=False)
def load_raw_data():
    df_m = pd.read_csv(MENSALIDADES_PATH)
    df_c = pd.read_csv(COBRANCAS_PATH)

    # Datas
    for col in ["data_competencia", "data_vencimento", "data_baixa"]:
        if col in df_m.columns:
            df_m[col] = pd.to_datetime(df_m[col], errors="coerce")
    df_c["data_cobranca"] = pd.to_datetime(df_c["data_cobranca"], errors="coerce")

    # Alvo binário: 1 se houve baixa
    df_m["foi_pago"] = np.where(df_m["data_baixa"].isna(), 0, 1)

    return df_m, df_c


@st.cache_data(show_spinner=False)
def build_action_dataset(max_rows: int = 250_000, random_state: int = 79) -> pd.DataFrame:
    """
    Build an action-level dataset:
      - une mensalidades e cobranças por id_aluno
      - keep collection actions up to 10 days after due date
      - para cada (aluno, data_cobranca, acao), associa a mensalidade mais próxima (em dias)
    """
    df_m, df_c = load_raw_data()

    mensalidades_base = df_m[["id_aluno", "data_vencimento", "valor_cobrado", "foi_pago"]].copy()
    cobrancas_base = df_c[["id_aluno", "acao_cobranca", "data_cobranca"]].copy()

    df_merged = pd.merge(mensalidades_base, cobrancas_base, on="id_aluno", how="inner")

    janela = pd.Timedelta(days=10)
    df_merged = df_merged[df_merged["data_cobranca"] <= (df_merged["data_vencimento"] + janela)].copy()

    df_merged["diferenca_data"] = (df_merged["data_cobranca"] - df_merged["data_vencimento"]).abs()
    df_merged = df_merged.sort_values("diferenca_data")

    df = df_merged.drop_duplicates(subset=["id_aluno", "data_cobranca", "acao_cobranca"]).copy()
    df["dias_dif"] = df["diferenca_data"].dt.days

    # Limpeza e seleção final
    df = df.dropna(subset=["acao_cobranca", "dias_dif", "foi_pago", "valor_cobrado"]).copy()
    df["foi_pago"] = df["foi_pago"].astype(int)

    # Amostragem para deixar o app rápido
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)

    # Normalize action strings to avoid duplicates due to capitalization
    df["acao_cobranca"] = df["acao_cobranca"].astype(str).str.strip()

    return df[["acao_cobranca", "dias_dif", "valor_cobrado", "foi_pago"]]


@st.cache_data(show_spinner=False)
def get_action_list(df: pd.DataFrame) -> list[str]:
    actions = sorted(df["acao_cobranca"].dropna().unique().tolist())
    return actions


# ---------------------------
# Modelo
# ---------------------------
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, use_calibration: bool = True, random_state: int = 79):
    X = df[["acao_cobranca", "dias_dif", "valor_cobrado"]]
    y = df["foi_pago"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    cat_features = ["acao_cobranca"]
    num_features = ["dias_dif", "valor_cobrado"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
    )

    # XGBoost: rápido + bom para tabular
    base_model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=random_state,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", base_model)])

    if use_calibration:
        # Calibration for a more “product-like” feel (more reliable probabilities)
        clf = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        return clf, {"AUC": auc, "Brier": brier}, (X_test, y_test, proba)
    else:
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        return pipe, {"AUC": auc, "Brier": brier}, (X_test, y_test, proba)


def predict_proba(model, df_inputs: pd.DataFrame) -> np.ndarray:
    # model pode ser CalibratedClassifierCV ou Pipeline
    return model.predict_proba(df_inputs)[:, 1]


def rank_actions(model, actions: list[str], dias_dif: int, valor_cobrado: float) -> pd.DataFrame:
    df_try = pd.DataFrame({
        "acao_cobranca": actions,
        "dias_dif": [dias_dif] * len(actions),
        "valor_cobrado": [valor_cobrado] * len(actions),
    })
    df_try["p_pagamento"] = predict_proba(model, df_try)
    df_try = df_try.sort_values("p_pagamento", ascending=False).reset_index(drop=True)
    return df_try


def best_action_by_roi(df_rank: pd.DataFrame, custos: dict[str, float], valor_cobrado: float) -> pd.DataFrame:
    df = df_rank.copy()
    df["custo"] = df["acao_cobranca"].map(custos).fillna(0.0)
    df["retorno_esperado"] = df["p_pagamento"] * float(valor_cobrado) - df["custo"]
    return df.sort_values("retorno_esperado", ascending=False).reset_index(drop=True)


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📲", layout="wide")
st.title("📲 Next Best Action for Collections")
st.caption("Ranks collection actions by predicted payment probability, with ROI simulation and explainability.")


with st.sidebar:
    st.header("Settings")
    max_rows = st.slider("Training sample size (rows)", 50_000, 400_000, 250_000, step=50_000)
    use_calib = st.toggle("Calibrate probabilities (recommended)", value=True)

    st.divider()
    st.subheader("Recommendation scenario")
    dias_dif = st.slider("days_past_due (days relative to due date)", 0, 10, 3, step=1)

    valor_cobrado_input = st.number_input("Invoice amount (BRL)", min_value=0.0, value=600.0, step=10.0)

    st.divider()
    st.subheader("Action costs (for ROI)")
    custo_default = {
        "SMS": 0.05,
        "whatsapp": 0.15,
        "e-mail": 0.03,
        "ligação telefônica": 3.50,
    }
    # O usuário pode editar custos mais comuns
    custo_sms = st.number_input("SMS cost (BRL)", min_value=0.0, value=float(custo_default["SMS"]), step=0.01)
    custo_wpp = st.number_input("WhatsApp cost (BRL)", min_value=0.0, value=float(custo_default["whatsapp"]), step=0.01)
    custo_email = st.number_input("Email cost (BRL)", min_value=0.0, value=float(custo_default["e-mail"]), step=0.01)
    custo_lig = st.number_input("Phone call cost (BRL)", min_value=0.0, value=float(custo_default["ligação telefônica"]), step=0.10)

    explain = st.toggle("Explainability (SHAP) — may be slow", value=False)
    if explain and not _HAS_SHAP:
        st.warning("SHAP is not installed/compatible in this environment. Turn this off or install `shap`.")


@st.cache_data(show_spinner=False)
def build_cost_map(actions: list[str], custo_sms, custo_wpp, custo_email, custo_lig):
    custos = {}
    for a in actions:
        key = str(a).strip().lower()
        if key == "sms":
            custos[a] = float(custo_sms)
        elif key == "whatsapp":
            custos[a] = float(custo_wpp)
        elif key in ["e-mail", "email", "e mail", "e_mail"]:
            custos[a] = float(custo_email)
        elif key in ["ligação telefônica", "ligacao telefonica", "ligacao", "ligação", "telefone"]:
            custos[a] = float(custo_lig)
        else:
            custos[a] = float(custo_wpp)  # fallback razoável
    return custos


# Carrega e treina
with st.spinner("Loading data and building action-level dataset..."):
    df_actions = build_action_dataset(max_rows=max_rows)
actions = get_action_list(df_actions)
custos_map = build_cost_map(actions, custo_sms, custo_wpp, custo_email, custo_lig)

with st.spinner("Training model..."):
    model, metrics, holdout = train_model(df_actions, use_calibration=use_calib)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("AUC (holdout)", f"{metrics['AUC']:.4f}")
with col2:
    st.metric("Brier score (holdout)", f"{metrics['Brier']:.4f}")
with col3:
    st.write("**Quick interpretation**: AUC measures ranking (higher is better). Brier measures calibration (lower is better).")


tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendation", "📈 Policy over time", "🧠 Explainability", "📦 Data"])

# --- Tab 1: recommendation
with tab1:
    st.subheader("Action ranking (Next Best Action)")
    st.write("Given **days_past_due** and **amount**, we compute the predicted payment probability for each action.")

    df_rank = rank_actions(model, actions, dias_dif=dias_dif, valor_cobrado=valor_cobrado_input)
    df_roi = best_action_by_roi(df_rank, custos_map, valor_cobrado_input)

    best_prob = df_rank.iloc[0]
    best_roi = df_roi.iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.success(f"**Best by probability:** {best_prob['acao_cobranca']}  —  {best_prob['p_pagamento']:.1%}")
        st.dataframe(df_rank.head(10), use_container_width=True)
    with c2:
        st.success(f"**Best by ROI:** {best_roi['acao_cobranca']}  —  expected return BRL {best_roi['retorno_esperado']:.2f}")
        st.dataframe(df_roi.head(10), use_container_width=True)

    st.caption("Video tip: move the days_past_due slider and watch the ranking update in real time.")

# --- Tab 2: política
with tab2:
    st.subheader("Simple policy: best action by ROI across days_past_due")
    st.write("We simulate **days_past_due = 0..10** and pick the action with the highest expected return.")

    rows = []
    for d in range(0, 11):
        df_rank_d = rank_actions(model, actions, dias_dif=d, valor_cobrado=valor_cobrado_input)
        df_roi_d = best_action_by_roi(df_rank_d, custos_map, valor_cobrado_input)
        rows.append({
            "dias_dif": d,
            "acao_otima": df_roi_d.loc[0, "acao_cobranca"],
            "p_pagamento": float(df_roi_d.loc[0, "p_pagamento"]),
            "retorno_esperado": float(df_roi_d.loc[0, "retorno_esperado"]),
        })
    pol = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        fig = plt.figure()
        plt.plot(pol["dias_dif"], pol["retorno_esperado"])
        plt.xlabel("dias_dif")
        plt.ylabel("retorno esperado (R$)")
        plt.title("Expected return of the optimal action by days_past_due")
        st.pyplot(fig)
    with c2:
        st.dataframe(pol, use_container_width=True)

# --- Tab 3: explainability
with tab3:
    st.subheader("Explainability")
    if not explain:
        st.info("Enable \"Explainability (SHAP)\" in the sidebar to view explanations (may be slow).")
    else:
        if not _HAS_SHAP:
            st.error("SHAP is unavailable. Install `shap` and restart the app.")
        else:
            X_test, y_test, proba = holdout
            st.write("We will explain an example from the holdout set.")
            idx = st.slider("Pick a holdout index", 0, min(200, len(X_test)-1), 0)
            x_row = X_test.iloc[[idx]].copy()
            p = float(predict_proba(model, x_row)[0])
            st.write("**Entrada**")
            st.dataframe(x_row, use_container_width=True)
            st.write(f"**Predicted payment probability:** {p:.1%}")

            # Acesso ao pipeline interno (dependendo se calibrado)
            # Para CalibratedClassifierCV, o estimador é model.base_estimator_ após fit (sklearn >=1.3),
            # e fica em model.calibrated_classifiers_[0].estimator em versões antigas.
            base = None
            try:
                base = model.base_estimator
            except Exception:
                try:
                    base = model.calibrated_classifiers_[0].estimator
                except Exception:
                    base = None

            if base is None:
                st.warning("Could not access the base pipeline for SHAP. Disable calibration or upgrade scikit-learn.")
            else:
                # transforma features
                X_trans = base.named_steps["prep"].transform(x_row)
                # pega o modelo xgb
                xgb = base.named_steps["model"]

                explainer = shap.TreeExplainer(xgb)
                shap_values = explainer.shap_values(X_trans)

                st.caption("SHAP explains the contribution of each encoded feature (including one-hot).")

                # Plot waterfall
                fig = plt.figure()
                shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                                     base_values=explainer.expected_value,
                                                     data=X_trans[0]),
                                     show=False)
                st.pyplot(fig)

# --- Tab 4: data
with tab4:
    st.subheader("Action-level dataset sample")
    st.dataframe(df_actions.head(30), use_container_width=True)
    st.write("**Target distribution (`foi_pago`):**")
    st.write(df_actions["foi_pago"].value_counts(normalize=True).rename("proportion"))