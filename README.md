# ⚡ PredictiveEnergy

Maintenance prédictive IIoT : détection de pannes imminentes à partir de données capteurs, avec une API de prédiction et un dashboard temps réel.

<img width="1917" height="956" alt="dashboard" src="https://github.com/user-attachments/assets/72c89a4f-b002-4a0e-9253-c9907cdb120e" />


## Fonctionnalités

- Prédiction de panne imminente avec un modèle **XGBoost** (66 features enrichies)
- **API FastAPI** (`/health`, `/predict`) avec seuil d'alerte à 0.65
- **Dashboard Streamlit** en temps réel : état de la machine, confiance, historique, taux d'alerte glissant
- Métriques live (recall, precision) calculées sur les dernières mesures
- Modes de simulation : séquentiel, pannes uniquement, normaux uniquement

## Architecture

```
Données de test (capteurs)
        │
        ▼
Dashboard Streamlit ──► API FastAPI ──► Scaler + XGBoost
        ▲                                     │
        └──────── probabilité + alerte ◄──────┘
```

## Structure du projet

```
Predictive_Energy/
├── api/            # API FastAPI (main.py, predictor.py, schemas.py)
├── dashboard/      # Dashboard Streamlit (app.py)
├── notebooks/      # Exploration, feature engineering, entraînement
├── models/         # Modèles et scalers (non versionnés)
├── data/           # Données (non versionnées)
├── docs/           # Captures d'écran
└── README.md
```

## Installation

```bash
git clone https://github.com/TON-USER/Predictive_Energy.git
cd Predictive_Energy
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Lancement

Dans deux terminaux séparés :

```bash
# Terminal 1 : API
cd api
python -m uvicorn main:app --reload --port 8000

# Terminal 2 : Dashboard
cd dashboard
streamlit run app.py
```

Le dashboard est accessible sur http://localhost:8501 et la documentation de l'API sur http://localhost:8000/docs.

## Données et modèles

Les dossiers `data/` et `models/` ne sont pas versionnés (taille des fichiers).
Pour les régénérer, exécuter les notebooks du dossier `notebooks/`.

## Résultats

| Métrique  | Valeur |
|-----------|--------|
| Recall    | à compléter |
| Precision | à compléter |
| F1-score  | à compléter |

## Stack technique

Python · XGBoost · scikit-learn · FastAPI · Streamlit · Plotly · pandas

## Auteur

**Adnane Ben Ammar**, étudiant ingénieur IA & Data Science, EPI Education Group
