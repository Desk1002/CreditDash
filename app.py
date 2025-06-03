from importlib.resources import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve 


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import os
import joblib
from pathlib import Path

import shap
from utils import convert_df
from eda_app import run_eda_app
#from config_app import run_config_app
from ml_app import run_ml_app
from eval_app import run_eval_app
from explain_app import run_explain_app
from validation_app import run_validation_app


import logging
import sys
import streamlit as st
import os
from pathlib import Path
import logging
import sys
import time


# Logging setup
logging.basicConfig(
    filename='logs.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(stdout_handler)

# Set paths
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)

def display_home():
    """Display the home page with operational action plan and risk pyramid."""
    # Custom CSS for consistent aesthetic
    st.markdown("""
        <style>
            .main-container {
                background: linear-gradient(135deg, #003087 0%, #E6F0FA 100%);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .logo-container {
                text-align: center;
                margin-bottom: 20px;
            }
            .section-title {
                color: #FFFFFF;
                font-size: 2.5em;
                font-weight: bold;
                text-align: center;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Main container
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Afriland First Bank Logo
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image(
            "https://tse3.mm.bing.net/th/id/OIP.CaSXc3j7SHpFh7sv0wVDFwHaE9?rs=1&pid=ImgDetMain",
            width=150,
            caption="Afriland First Bank"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Welcome title
        st.markdown('<p class="section-title">Customer Analytics for Credit Risk</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Plan d'Action Opérationnel
    st.markdown("""
        <div style="border: 2px solid #28B463; padding: 15px; border-radius: 10px; font-family: Arial, sans-serif; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="margin: 0; color: #28B463; text-align: center; font-size: 1.8em; font-weight: bold;">
                ⚙️ PLAN D'ACTION OPÉRATIONNEL ⚙️
            </h3>
            <div style="display: flex; justify-content: space-around; margin-top: 20px; text-align: center;">
                <div style="width: 30%;">
                    <div style="font-size: 3.5em; line-height: 1;">➡️</div>
                    <div style="background-color: #F4F6F7; border: 1px solid #D5D8DC; padding: 8px 5px; border-radius: 5px;">
                        <strong style="font-size: 1.2em;">1. SCORING AUTOMATIQUE</strong><br>
                        <small style="font-size: 0.85em;">Chaque crédit du portefeuille reçoit une probabilité de défaut actualisée.</small>
                    </div>
                </div>
                <div style="width: 30%;">
                    <div style="font-size: 3.5em; line-height: 1;">➡️</div>
                    <div style="background-color: #F4F6F7; border: 1px solid #D5D8DC; padding: 8px 5px; border-radius: 5px;">
                        <strong style="font-size: 1.2em;">2. TRIAGE PAR NIVEAU DE RISQUE</strong><br>
                        <small style="font-size: 0.85em;">Le portefeuille est segmenté en 3 niveaux d'alerte.</small>
                    </div>
                </div>
                <div style="width: 30%;">
                    <div style="font-size: 3.5em; line-height: 1;">➡️</div>
                    <div style="background-color: #F4F6F7; border: 1px solid #D5D8DC; padding: 8px 5px; border-radius: 5px;">
                        <strong style="font-size: 1.2em;">3. ACTIONS DIFFÉRENCIÉES</strong><br>
                        <small style="font-size: 0.85em;">Chaque niveau de risque déclenche une réponse adaptée.</small>
                    </div>
                </div>
            </div>
            <div style="margin-top: 25px;">
                <div style="background-color: #C70039; color: white; padding: 12px 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong style="font-size: 1.3em;">RISQUE ÉLEVÉ (Score > 0.75)</strong> : <span style="font-size: 1.1em;">📞 Intervention immédiate, contact client, plan de restructuration.</span>
                </div>
                <div style="background-color: #FFC300; color: #333; padding: 12px 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong style="font-size: 1.3em;">RISQUE MODÉRÉ (Score 0.40 - 0.75)</strong> : <span style="font-size: 1.1em;">📊 Placement sous surveillance renforcée, suivi mensuel des indicateurs clés.</span>
                </div>
                <div style="background-color: #28B463; color: white; padding: 12px 10px; border-radius: 5px;">
                    <strong style="font-size: 1.3em;">RISQUE FAIBLE (Score < 0.40)</strong> : <span style="font-size: 1.1em;">✅ Suivi standard du portefeuille.</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # La Pyramide du Risque
    st.markdown("""
        <div style="border: 2px solid #C70039; padding: 15px; border-radius: 10px; text-align: center; font-family: Arial, sans-serif;">
            <h3 style="margin: 0; color: #C70039;">🔺 LA PYRAMIDE DU RISQUE 🔺</h3>
            <p><em>Quels facteurs prédisent un impayé ?</em></p>
            <div style="font-size: 24px; font-weight: bold; color: #C70039; margin-top: 20px;">
                ⚠️ NIVEAU D'ALERTE MAXIMAL ⚠️
            </div>
            <div style="background-color: #FF5733; color: white; padding: 10px; margin-top: 10px; border-radius: 5px;">
                <strong>Comportement de Paiement Immédiat</strong><br>
                <small><em>Signaux critiques nécessitant une action immédiate.</em></small><br>
                ▶️ nb_impayes_cumul : Le client a-t-il déjà des impayés ? C'est le signal N°1. <br>
                ▶️ jours_depuis_dernier_incident : Un incident récent est un drapeau rouge vif.
            </div>
            <div style="font-size: 20px; font-weight: bold; color: #FF5733; margin-top: 15px;">
                🔥 NIVEAU D'ALERTE ÉLEVÉ 🔥
            </div>
            <div style="background-color: #FFC300; color: #333; padding: 10px; margin-top: 10px; border-radius: 5px;">
                <strong>Tendances Comportementales Récentes</strong><br>
                <small><em>Indicateurs de dégradation de la santé financière.</em></small><br>
                ▶️ retard_moyen_pass_short_win : Le client commence-t-il à payer systématiquement en retard ?<br>
                ▶️ ratio_paiement_moyen_pass_short_win : La proportion de ses paiements honorés diminue-t-elle ?
            </div>
            <div style="font-size: 18px; font-weight: bold; color: #FFC300; margin-top: 15px;">
                🔍 NIVEAU DE SURVEILLANCE 🔍
            </div>
            <div style="background-color: #DAF7A6; color: #333; padding: 10px; margin-top: 10px; border-radius: 5px;">
                <strong>Caractéristiques Structurelles et Contextuelles</strong><br>
                <small><em>Facteurs de risque de fond à surveiller.</em></small><br>
                ▶️ Caractéristiques du prêt : capital Restant Du, NbreEch.<br>
                ▶️ Profil du client : Type de Credit, FormeJuridique_EURL, Region_North.
            </div>
        </div>
    """, unsafe_allow_html=True)

    logging.info("Displayed home page with operational action plan and risk pyramid.")

def main():
    """Main function to control app navigation."""
    # Initialize session state
    if 'menu_choice' not in st.session_state:
        st.session_state.menu_choice = "Home"

    # Sidebar navigation
    st.sidebar.title("Afriland First Bank Analytics")
    menu = ["Home", "About", "EDA", "ML", "Eval", "Explain", "Validation"]
    st.session_state.menu_choice = st.sidebar.selectbox(
        "Menu", menu, index=menu.index(st.session_state.menu_choice)
    )

    # Route to appropriate module
    try:
        if st.session_state.menu_choice == "Home":
            display_home()
        elif st.session_state.menu_choice == "About":
            st.subheader("About")
            try:
                st.markdown(Path('About.md').read_text())
            except FileNotFoundError:
                st.error("About.md not found.")
                logging.error("About.md file not found.")
        elif st.session_state.menu_choice == "EDA":
            st.subheader('Exploratory Data Analysis')
            run_eda_app()
            logging.info("Running EDA app.")
        elif st.session_state.menu_choice == "ML":
            st.subheader("Machine Learning")
            run_ml_app()
            logging.info("Running ML app.")
        elif st.session_state.menu_choice == "Eval":
            st.subheader("Model Evaluation")
            run_eval_app()
            logging.info("Running Eval app.")
        elif st.session_state.menu_choice == "Explain":
            st.subheader("Model Explanation")
            run_explain_app()
            logging.info("Running Explain app.")
        elif st.session_state.menu_choice == "Validation":
            st.subheader("Model Validation")
            run_validation_app()
            logging.info("Running Validation app.")
    except Exception as e:
        st.error(f"Error in {st.session_state.menu_choice} module: {e}")
        logging.error(f"Error in {st.session_state.menu_choice} module: {e}")

if __name__ == "__main__":
    main()