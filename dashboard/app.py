# ================================================================
# Dashboard Streamlit — PredictiveEnergy (sans Telegram)
# Compatible API AVEC ou SANS authentification (/login)
# Fichier : dashboard/app.py
# Lancer  : streamlit run app.py        (Streamlit >= 1.37 requis)
# ================================================================

import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ================================================================
# CONFIGURATION PAGE
# ================================================================

st.set_page_config(
    page_title="PredictiveEnergy",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CONFIGURATION
# ================================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")

BASE = Path(__file__).resolve().parent.parent
SCALER_PATH = BASE / "models" / "scaler_enrichi.pkl"
XTEST_PATH = BASE / "data" / "processed" / "X_test_enrichi.csv"
YTEST_PATH = BASE / "data" / "processed" / "y_test_enrichi.csv"

# ================================================================
# CSS
# ================================================================

st.markdown(
    """
<style>
    .alert-box {
        background: #3D1515; border: 2px solid #FF4444;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .normal-box {
        background: #153D1E; border: 2px solid #44FF88;
        border-radius: 12px; padding: 20px; text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ================================================================
# FONCTIONS
# ================================================================


@st.cache_resource
def charger_ressources():
    scaler = joblib.load(SCALER_PATH)
    X_test = pd.read_csv(XTEST_PATH)
    y_test = pd.read_csv(YTEST_PATH).squeeze()
    return scaler, X_test, y_test


def verifier_api():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def api_a_auth():
    """True si l'API expose une route /login (sinon pas d'authentification)."""
    try:
        paths = requests.get(f"{API_URL}/openapi.json", timeout=3).json()["paths"]
        return "/login" in paths
    except (requests.RequestException, KeyError, ValueError):
        return False


def obtenir_token(username, password):
    try:
        resp = requests.post(
            f"{API_URL}/login",
            json={"username": username, "password": password},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()["access_token"], None
        if resp.status_code == 401:
            return None, "Identifiants incorrects"
        return None, f"Erreur {resp.status_code} : {resp.text[:200]}"
    except requests.RequestException as e:
        return None, f"API non accessible : {e}"


def predire(token, payload):
    """Retourne (resultat, erreur, status_code). Token optionnel."""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json(), None, 200
        return None, f"Erreur {resp.status_code} : {resp.text[:300]}", resp.status_code
    except requests.RequestException as e:
        return None, str(e), None


def construire_payload(scaler, X_test, pos):
    ligne_raw = pd.DataFrame(
        scaler.inverse_transform(X_test.iloc[[pos]]), columns=X_test.columns
    ).iloc[0]
    payload = {col: float(val) for col, val in ligne_raw.items()}
    for champ in ("courant", "temperature", "vibration"):
        payload.setdefault(champ, 0.0)
    payload["device_id"] = "ESP32_001"
    return payload


def reset_all():
    st.session_state.historique = []
    st.session_state.nb_alertes = 0
    st.session_state.nb_normaux = 0
    st.session_state.idx_courant = 0
    st.session_state.simulation_on = False


def deconnecter():
    st.session_state.token = None
    st.session_state.connecte = False
    st.session_state.username = None
    st.session_state.simulation_on = False


# ================================================================
# SESSION STATE
# ================================================================

defaults = {
    "token": None,
    "connecte": False,
    "username": None,
    "historique": [],
    "simulation_on": False,
    "idx_courant": 0,
    "nb_alertes": 0,
    "nb_normaux": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- API sans authentification : connexion automatique ----
AUTH_REQUISE = api_a_auth()
if not AUTH_REQUISE and not st.session_state.connecte:
    st.session_state.connecte = True
    st.session_state.username = "sans authentification"

# ================================================================
# CHARGEMENT RESSOURCES
# ================================================================

try:
    scaler, X_test, y_test = charger_ressources()
except Exception as e:
    st.error(f"Impossible de charger les ressources : {e}")
    st.stop()

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("## ⚡ PredictiveEnergy")
    st.markdown("---")

    if verifier_api():
        st.success("API connectee")
    else:
        st.error("API non accessible")
        st.info("Lance : uvicorn main:app --reload --port 8000")

    st.markdown("---")

    # ---- Connexion ----
    st.markdown("### Connexion")
    if not AUTH_REQUISE:
        st.info("API sans authentification : acces direct.")
    elif st.session_state.connecte:
        st.success(f"Connecte : {st.session_state.username}")
        if st.button("Se deconnecter", use_container_width=True):
            deconnecter()
            st.rerun()
    else:
        with st.form("login_form"):
            username = st.selectbox("Utilisateur", ["admin", "reader", "sensor"])
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter", use_container_width=True)
        if submit:
            token, erreur = obtenir_token(username, password)
            if token:
                st.session_state.token = token
                st.session_state.connecte = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error(erreur)

    st.markdown("---")

    # ---- Paramètres ----
    st.markdown("### Parametres")
    vitesse = st.slider("Vitesse (s)", 0.5, 3.0, 1.0, 0.5)
    nb_mesures_max = st.slider("Historique max", 20, 100, 50)
    seuil_alerte = st.slider("Seuil alerte (%)", 30, 80, 50, 5) / 100
    utiliser_seuil = st.checkbox(
        "Appliquer ce seuil (sinon decision de l'API)",
        value=False,
        help="A cocher seulement si 'confidence' = probabilite de PANNE.",
    )

    st.markdown("---")
    if st.button("Reinitialiser tout", use_container_width=True):
        reset_all()
        st.rerun()

# ================================================================
# PAGE PRINCIPALE
# ================================================================

st.markdown("# ⚡ PredictiveEnergy Dashboard")
st.markdown("*Maintenance Predictive IIoT — Detection de pannes en temps reel*")
st.markdown("---")

if not st.session_state.connecte:
    st.info("Connecte-toi dans la barre laterale pour commencer.")
    st.stop()

# ================================================================
# CONTROLES
# ================================================================

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("▶ Demarrer", use_container_width=True, type="primary",
                 disabled=st.session_state.simulation_on):
        st.session_state.simulation_on = True
        st.rerun()
with col2:
    if st.button("⏹ Arreter", use_container_width=True,
                 disabled=not st.session_state.simulation_on):
        st.session_state.simulation_on = False
        st.rerun()
with col3:
    if st.button("↺ Reinitialiser", use_container_width=True):
        reset_all()
        st.rerun()
with col4:
    mode = st.selectbox("Mode", ["Sequentiel", "Pannes uniquement", "Normaux uniquement"])

st.markdown("---")

# ================================================================
# MODE -> INDICES DISPONIBLES
# ================================================================

if mode == "Pannes uniquement":
    indices_dispo = list(y_test[y_test == 1].index)
elif mode == "Normaux uniquement":
    indices_dispo = list(y_test[y_test == 0].index)
else:
    indices_dispo = list(y_test.index)

if not indices_dispo:
    st.warning(f"Aucune donnee disponible pour le mode « {mode} ».")
    st.stop()

# ================================================================
# ETAPE DE SIMULATION
# ================================================================


def etape_simulation():
    idx_pos = st.session_state.idx_courant % len(indices_dispo)
    idx_reel = indices_dispo[idx_pos]
    pos = y_test.index.get_loc(idx_reel)

    payload = construire_payload(scaler, X_test, pos)
    resultat, erreur, code = predire(st.session_state.token, payload)

    if code == 401 and AUTH_REQUISE:
        deconnecter()
        st.warning("Session expiree, reconnecte-toi.")
        st.rerun()
        return
    if erreur:
        st.session_state.simulation_on = False
        st.error(f"Erreur API : {erreur}")
        return

    mesure_num = st.session_state.idx_courant + 1
    # L'API renvoie "probabilite" / "alerte" (anciens noms : confidence / alert)
    confidence = float(resultat.get("probabilite", resultat.get("confidence", 0.0)))
    alerte_api = resultat.get("alerte", resultat.get("alert", False))
    alerte = (confidence >= seuil_alerte) if utiliser_seuil else bool(alerte_api)

    st.session_state.historique.append({
        "temps": datetime.now().strftime("%H:%M:%S"),
        "mesure": mesure_num,
        "prediction": int(alerte),
        "confidence": confidence,
        "alert": alerte,
        "label_reel": int(y_test.loc[idx_reel]),
    })
    st.session_state.historique = st.session_state.historique[-nb_mesures_max:]

    if alerte:
        st.session_state.nb_alertes += 1
    else:
        st.session_state.nb_normaux += 1
    st.session_state.idx_courant += 1


# ================================================================
# AFFICHAGE
# ================================================================


def afficher():
    hist = st.session_state.historique
    if not hist:
        st.info("Clique sur « Demarrer » pour lancer la simulation.")
        return

    derniere = hist[-1]

    if derniere["alert"]:
        st.markdown(
            """<div class="alert-box">
            <h1 style="color:#FF4444;margin:0">🚨 PANNE IMMINENTE</h1>
            <p style="color:#FF8888;font-size:1.2rem;margin:8px 0">
                Intervention requise</p></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="normal-box">
            <h1 style="color:#44FF88;margin:0">✅ MACHINE EN BON ETAT</h1>
            <p style="color:#88FFAA;font-size:1.2rem;margin:8px 0">
                Fonctionnement normal</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction", "PANNE" if derniere["alert"] else "Normal")
    c2.metric("Confiance", f"{derniere['confidence']*100:.1f}%")
    c3.metric("Mesure n°", derniere["mesure"])
    correct = derniere["prediction"] == derniere["label_reel"]
    c4.metric("Correct ?", "✓ Oui" if correct else "✗ Non")

    df = pd.DataFrame(hist)
    tp = int(((df["prediction"] == 1) & (df["label_reel"] == 1)).sum())
    fp = int(((df["prediction"] == 1) & (df["label_reel"] == 0)).sum())
    fn = int(((df["prediction"] == 0) & (df["label_reel"] == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None

    total = st.session_state.nb_alertes + st.session_state.nb_normaux
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total mesures", total)
    s2.metric("Alertes", st.session_state.nb_alertes)
    s3.metric("Taux alerte",
              f"{st.session_state.nb_alertes/total*100:.1f}%" if total else "—")
    s4.metric("Recall (fenetre)", f"{recall*100:.1f}%" if recall is not None else "—")
    s5.metric("Precision (fenetre)",
              f"{precision*100:.1f}%" if precision is not None else "—")

    if len(df) >= 2:
        g1, g2 = st.columns(2)
        with g1:
            couleurs = ["#FF4444" if a else "#44FF88" for a in df["alert"]]
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df["mesure"], y=df["confidence"] * 100, mode="lines+markers",
                line=dict(color="#7C6AF5", width=2),
                marker=dict(color=couleurs, size=8), name="Confiance"))
            fig1.add_hline(y=seuil_alerte * 100, line_dash="dash",
                           line_color="#FF8800",
                           annotation_text=f"Seuil {seuil_alerte*100:.0f}%")
            fig1.update_layout(
                title="Confiance de prediction", xaxis_title="Mesure",
                yaxis_title="Confiance (%)", yaxis_range=[0, 105],
                template="plotly_dark", height=300,
                margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig1, use_container_width=True)

        with g2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df["mesure"], y=df["prediction"],
                marker_color=["#FF4444" if a else "#44FF88" for a in df["alert"]]))
            fig2.update_layout(
                title="Historique predictions (0=Normal | 1=Panne)",
                xaxis_title="Mesure", yaxis_title="Prediction",
                yaxis=dict(tickvals=[0, 1], ticktext=["Normal", "Panne"]),
                template="plotly_dark", height=300,
                margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig2, use_container_width=True)

        if len(df) >= 5:
            df["taux"] = df["alert"].astype(int).rolling(5, min_periods=1).mean() * 100
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df["mesure"], y=df["taux"], fill="tozeroy",
                fillcolor="rgba(255,68,68,0.15)",
                line=dict(color="#FF4444", width=2), name="Taux alerte glissant"))
            fig3.add_hline(y=30, line_dash="dash", line_color="#FF8800",
                           annotation_text="Seuil critique 30%")
            fig3.update_layout(
                title="Taux d'alerte glissant (fenetre 5 mesures)",
                xaxis_title="Mesure", yaxis_title="Taux (%)",
                yaxis_range=[0, 105], template="plotly_dark", height=250,
                margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Historique des 10 dernieres mesures")
    dd = df.tail(10).copy()
    dd["statut"] = dd["alert"].map({True: "🚨 ALERTE", False: "✅ Normal"})
    dd["confiance"] = (dd["confidence"] * 100).round(1).astype(str) + "%"
    dd["correct"] = (dd["prediction"] == dd["label_reel"]).map({True: "✓", False: "✗"})
    st.dataframe(dd[["mesure", "temps", "statut", "confiance", "correct"]],
                 use_container_width=True, hide_index=True)


# ================================================================
# BOUCLE LIVE (fragment : seule cette zone se rafraichit)
# ================================================================


if st.session_state.simulation_on:
    etape_simulation()
    st.caption(f"Simulation en cours… mesure n° {st.session_state.idx_courant}")

afficher()

if st.session_state.simulation_on:
    time.sleep(vitesse)
    st.rerun()