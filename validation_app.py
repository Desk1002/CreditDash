import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils import load, convert_df
import csv

# Set paths
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)

# Custom CSS
st.markdown("""
<style>
    .dashboard-container {
        border: 2px solid #003087;
        border-radius: 10px;
        padding: 20px;
        background-color: #F8F9FA;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #E6F0FA;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 1.8em;
        color: #003087;
        margin-top: 20px;
    }
    .alert-high {
        background-color: #C70039;
        color: white;
        padding: 12px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .alert-moderate {
        background-color: #FFC300;
        color: #333;
        padding: 12px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .alert-low {
        background-color: #28B463;
        color: white;
        padding: 12px;
        border-radius: 5px;
    }
    .pyramid-container {
        border: 2px solid #C70039;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-family: Arial, sans-serif;
        margin-top: 20px;
    }
    .action-plan {
        border: 2px solid #28B463;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def convert_df(df):
    """Convert DataFrame to CSV format for download."""
    return df.to_csv(index=False, quoting=csv.QUOTE_ALL).encode('utf-8')

def load_data(df, params):
    """Preprocess validation data based on critical parameters."""
    df = df.copy()
    
    # Apply date filter
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'] <= params['DATE_ARRETE_DONNEES']]
    
    # Flag critical overdue
    if 'retard_jours_effectif' in df.columns:
        df['Critical_Overdue'] = df['retard_jours_effectif'] > params['SEUIL_RETARD_CRITIQUE_JOURS']
    
    # Flag incomplete payment
    if 'ratio_paiement_effectif' in df.columns:
        df['Incomplete_Payment'] = df['ratio_paiement_effectif'] < params['SEUIL_PAIEMENT_INCOMPLET_RATIO']
    
    # Behavioral indicators (short/long window)
    for col in ['nb_incidents_paiement_pass_short_win', 'retard_moyen_pass_short_win', 'ratio_paiement_moyen_pass_short_win']:
        if col in df.columns:
            df[f'{col}_Alert'] = df[col] > df[col].quantile(0.75)  # Top 25% as alert
    
    return df

def run_validation_app():
    submenu = ["Prédiction des Probabilités", "Résumé des Risques"]
    choice = st.sidebar.selectbox("Sous-Menu", submenu)

    if choice == "Prédiction des Probabilités":
        st.subheader('Validation des Probabilités de Défaut')
        with st.container():
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            
            # Critical Parameters
            st.markdown('<p class="section-title">⚙️ Paramètres Critiques</p>', unsafe_allow_html=True)
            date_arrete_donnees = st.date_input("Date d'arrêt des données", value=datetime(2025, 4, 20))
            seuil_retard_critique_jours = st.slider("Seuil de retard critique (jours)", 0, 90, 0)
            seuil_paiement_incomplet_ratio = st.slider("Seuil de paiement incomplet (ratio)", 0.0, 1.0, 1.0, 0.05)
            window_behavior_short = st.slider("Fenêtre courte (échéances)", 1, 10, 3)
            window_behavior_long = st.slider("Fenêtre longue (échéances)", 5, 30, 20)
            
            params = {
                'DATE_ARRETE_DONNEES': pd.to_datetime(date_arrete_donnees),
                'SEUIL_RETARD_CRITIQUE_JOURS': seuil_retard_critique_jours,
                'SEUIL_PAIEMENT_INCOMPLET_RATIO': seuil_paiement_incomplet_ratio,
                'WINDOW_BEHAVIOR_SHORT': window_behavior_short,
                'WINDOW_BEHAVIOR_LONG': window_behavior_long
            }

            # Upload validation CSV
            uploaded_file = st.file_uploader("Charger le fichier CSV de validation", type=['csv'])
            delimiter = st.selectbox("Sélectionner le délimiteur", [",", ";", "\t", " "], index=0)
            encoding = st.selectbox("Sélectionner l'encodage", ["utf-8", "latin1", "iso-8859-1", "cp1252"], index=0)
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding, on_bad_lines='warn')
                    df = load(df, date_arrete_donnees, seuil_retard_critique_jours, seuil_paiement_incomplet_ratio,
                          window_behavior_short, window_behavior_long)

                    st.write(f"Données chargées avec {df.shape[0]} lignes et {df.shape[1]} colonnes.")
                    st.dataframe(df.head())
                    logging.info(f"Loaded validation dataset with {df.shape[0]} rows.")
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du CSV : {e}")
                    st.info("Essayez d'ajuster le délimiteur ou l'encodage.")
                    logging.error(f"Error reading validation CSV: {e}")
                    return

                # Model selection
                st.markdown('<p class="section-title">📊 Sélection du Modèle</p>', unsafe_allow_html=True)
                model_choice = st.radio("Choisir le modèle", ["Meilleur Modèle (Basé sur Recall)", "Sélection Manuelle"])
                pipeline = None
                
                if model_choice == "Meilleur Modèle (Basé sur Recall)":
                    try:
                        pipeline = joblib.load(MODELPATH / 'best_pipeline.pkl')
                        st.write("Meilleur modèle chargé depuis `best_pipeline.pkl`.")
                    except FileNotFoundError:
                        st.error("Meilleur modèle non trouvé. Veuillez entraîner les modèles dans l'onglet 'Best Model'.")
                        return
                else:
                    classifiers = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "ANN"]
                    selected_classifier = st.selectbox("Sélectionner le classificateur", classifiers)
                    try:
                        pipelines = joblib.load(MODELPATH / 'segmented_pipelines.pkl')
                        for key, p in pipelines.items():
                            if p.named_steps['classifier'].__class__.__name__ == selected_classifier:
                                pipeline = p
                                st.write(f"Modèle {selected_classifier} chargé.")
                                break
                        if pipeline is None:
                            st.error("Modèle non trouvé dans les pipelines segmentés.")
                            return
                    except FileNotFoundError:
                        st.error("Pipelines segmentés non trouvés. Veuillez entraîner les modèles segmentés.")
                        return

                # Predict probabilities
                if st.button('Prédire les Probabilités de Défaut'):
                    try:
                        features = joblib.load(MODELPATH / 'num_features.pkl') + [c for c in joblib.load(MODELPATH / 'cat_features.pkl') if c not in ['Type de Credit_G', 'Segment_G']]
                        missing_cols = [col for col in features if col not in df.columns]
                        if missing_cols:
                            st.error(f"Colonnes manquantes : {missing_cols}")
                            return
                        
                        df['Impayé_Probabilité'] = pipeline.predict_proba(df[features])[:, 1]
                        st.write("Données validées avec probabilités de défaut :")
                        st.dataframe(df)
                        
                        # Save predictions
                        joblib.dump(df, DATAPATH / 'validation_results.pkl')
                        csv = convert_df(df)
                        st.download_button(
                            label="Télécharger les prédictions sous forme de CSV",
                            data=csv,
                            file_name='validation_predictions.csv',
                            mime='text/csv'
                        )
                        logging.info("Generated validation predictions.")
                    except Exception as e:
                        st.error(f"Erreur lors des prédictions : {e}")
                        logging.error(f"Error during validation predictions: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "Résumé des Risques":
        st.subheader('Résumé des Risques')
        with st.container():
            st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
            
            try:
                df = joblib.load(DATAPATH / 'validation_results.pkl')
                df = load(df, date_arrete_donnees, seuil_retard_critique_jours, seuil_paiement_incomplet_ratio,
                          window_behavior_short, window_behavior_long)
                st.write(f"Données de validation chargées avec {df.shape[0]} lignes.")
            except FileNotFoundError:
                st.error("Aucune donnée validée trouvée. Veuillez d'abord prédire les probabilités.")
                return

            # Define alert thresholds
            st.markdown('<p class="section-title">⚠️ Seuils d’Alerte</p>', unsafe_allow_html=True)
            seuil_eleve = st.slider("Seuil pour Risque Élevé", 0.5, 1.0, 0.75, 0.05)
            seuil_moderé = st.slider("Seuil pour Risque Modéré", 0.2, 0.7, 0.40, 0.05)

            # Categorize risks
            df['Niveau_Risque'] = pd.cut(
                df['Impayé_Probabilité'],
                bins=[0, seuil_moderé, seuil_eleve, 1.0],
                labels=['Faible', 'Modéré', 'Élevé'],
                include_lowest=True
            )

            # Pie Chart
            st.markdown('<p class="section-title">📊 Répartition des Niveaux de Risque</p>', unsafe_allow_html=True)
            risk_counts = df['Niveau_Risque'].value_counts().reset_index()
            risk_counts.columns = ['Niveau_Risque', 'Nombre']
            fig = px.pie(risk_counts, names='Niveau_Risque', values='Nombre',
                         title='Répartition des Crédits par Niveau de Risque',
                         color='Niveau_Risque',
                         color_discrete_map={'Élevé': '#C70039', 'Modéré': '#FFC300', 'Faible': '#28B463'})
            st.plotly_chart(fig, use_container_width=True)

            # Critical Variables Alerts
            st.markdown('<p class="section-title">🚨 Alertes sur les Variables Critiques</p>', unsafe_allow_html=True)
            critical_vars = ['nb_impayes_cumul', 'jours_depuis_dernier_incident', 'retard_moyen_pass_short_win', 'ratio_paiement_moyen_pass_short_win']
            for var in critical_vars:
                if var in df.columns:
                    seuil_alert = st.slider(f"Seuil d’alerte pour {var}", float(df[var].min()), float(df[var].max()), float(df[var].quantile(0.75)))
                    high_risk = df[df[var] > seuil_alert]
                    if not high_risk.empty:
                        st.markdown(f'<div class="metric-card"><p class="metric-title">{var}</p><p class="metric-value">⚠️ {len(high_risk)} crédits dépassent le seuil ({seuil_alert:.2f})</p></div>', unsafe_allow_html=True)
                        fig = px.histogram(high_risk, x=var, title=f'Distribution de {var} (au-dessus du seuil)',
                                           color_discrete_sequence=['#C70039'])
                        st.plotly_chart(fig, use_container_width=True)

            # Pyramide du Risque
            st.markdown('<p class="section-title">🔺 Pyramide du Risque</p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="pyramid-container">
                <h3 style="color: #C70039;">🔺 LA PYRAMIDE DU RISQUE 🔺</h3>
                <p><em>Quels facteurs prédisent un impayé ?</em></p>
                <div style="font-size: 24px; font-weight: bold; color: #C70039; margin-top: 20px;">
                    ⚠️ NIVEAU D'ALERTE MAXIMAL ⚠️
                </div>
                <div style="background-color: #FF5733; color: white; padding: 10px; margin-top: 10px; border-radius: 5px;">
                    <strong>Comportement de Paiement Immédiat</strong><br>
                    <small><em>Signaux critiques nécessitant une action immédiate.</em></small><br>
                    ▶️ <strong>nb_impayes_cumul</strong> : {nb_impayes} crédits avec impayés cumulés.<br>
                    ▶️ <strong>jours_depuis_dernier_incident</strong> : {incidents_recentes} crédits avec incidents récents.
                </div>
                <div style="font-size: 20px; font-weight: bold; color: #FF5733; margin-top: 15px;">
                    🔥 NIVEAU D'ALERTE ÉLEVÉ 🔥
                </div>
                <div style="background-color: #FFC300; color: #333; padding: 10px; margin-top: 10px; border-radius: 5px;">
                    <strong>Tendances Comportementales Récentes</strong><br>
                    <small><em>Indicateurs de dégradation de la santé financière.</em></small><br>
                    ▶️ <strong>retard_moyen_pass_short_win</strong> : {retards_recentes} crédits avec retards moyens élevés.<br>
                    ▶️ <strong>ratio_paiement_moyen_pass_short_win</strong> : {paiements_faibles} crédits avec faible ratio de paiement.
                </div>
                <div style="font-size: 18px; font-weight: bold; color: #FFC300; margin-top: 15px;">
                    🔍 NIVEAU DE SURVEILLANCE 🔍
                </div>
                <div style="background-color: #DAF7A6; color: #333; padding: 10px; margin-top: 10px; border-radius: 5px;">
                    <strong>Caractéristiques Structurelles</strong><br>
                    <small><em>Facteurs de risque de fond à surveiller.</em></small><br>
                    ▶️ <strong>capital Restant Du, NbreEch</strong> : Surveillez les crédits à fort capital restant.<br>
                    ▶️ <strong>Type de Credit_G, Segment_G</strong> : Certains profils sont plus risqués.
                </div>
            </div>
            """.format(
                nb_impayes=len(df[df['nb_impayes_cumul'] > df['nb_impayes_cumul'].quantile(0.75)]) if 'nb_impayes_cumul' in df.columns else 0,
                incidents_recentes=len(df[df['jours_depuis_dernier_incident'] < df['jours_depuis_dernier_incident'].quantile(0.25)]) if 'jours_depuis_dernier_incident' in df.columns else 0,
                retards_recentes=len(df[df['retard_moyen_pass_short_win'] > df['retard_moyen_pass_short_win'].quantile(0.75)]) if 'retard_moyen_pass_short_win' in df.columns else 0,
                paiements_faibles=len(df[df['ratio_paiement_moyen_pass_short_win'] < df['ratio_paiement_moyen_pass_short_win'].quantile(0.25)]) if 'ratio_paiement_moyen_pass_short_win' in df.columns else 0
            ), unsafe_allow_html=True)

            # Plan d'Action
            st.markdown('<p class="section-title">⚙️ Plan d’Action Opérationnel</p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="action-plan">
                <h3 style="color: #28B463; text-align: center;">⚙️ PLAN D'ACTION OPÉRATIONNEL ⚙️</h3>
                <div style="display: flex; justify-content: space-around; margin-top: 20px; text-align: center;">
                    <div style="width: 30%;">
                        <div style="font-size: 3.5em;">➡️</div>
                        <div style="background-color: #F4F6F7; border: 1px solid #D5D8DC; padding: 8px; border-radius: 5px;">
                            <strong>1. SCORING AUTOMATIQUE</strong><br>
                            <small>{nb_credits} crédits scorés avec probabilités de défaut.</small>
                        </div>
                    </div>
                    <div style="width: 30%;">
                        <div style="font-size: 3.5em;">➡️</div>
                        <div style="background-color: #F4F6F7; border: 1px solid #D5D8DC; padding: 8px; border-radius: 5px;">
                            <strong>2. TRIAGE PAR RISQUE</strong><br>
                            <small>{nb_high_risk} crédits à risque élevé, {nb_moderate_risk} modérés, {nb_low_risk} faibles.</small>
                        </div>
                    </div>
                    <div style="width: 30%;">
                        <div style="font-size: 3.5em;">➡️</div>
                        <div style="background-color: #F4F6F7; border: 1px solid #D5D8DC; padding: 8px; border-radius: 5px;">
                            <strong>3. ACTIONS DIFFÉRENCIÉES</strong><br>
                            <small>Interventions adaptées à chaque niveau de risque.</small>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 25px;">
                    <div class="alert-high">
                        <strong>RISQUE ÉLEVÉ (Score > {seuil_eleve})</strong> : 📞 Intervention immédiate, contact client, restructuration.
                    </div>
                    <div class="alert-moderate">
                        <strong>RISQUE MODÉRÉ (Score {seuil_moderé} - {seuil_eleve})</strong> : 📊 Surveillance renforcée, suivi mensuel.
                    </div>
                    <div class="alert-low">
                        <strong>RISQUE FAIBLE (Score < {seuil_moderé})</strong> : ✅ Suivi standard.
                    </div>
                </div>
            </div>
            """.format(
                nb_credits=len(df),
                nb_high_risk=len(df[df['Niveau_Risque'] == 'Élevé']),
                nb_moderate_risk=len(df[df['Niveau_Risque'] == 'Modéré']),
                nb_low_risk=len(df[df['Niveau_Risque'] == 'Faible']),
                seuil_eleve=seuil_eleve,
                seuil_moderé=seuil_moderé
            ), unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

