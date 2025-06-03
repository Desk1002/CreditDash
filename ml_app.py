import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, f1_score, precision_score
import joblib
from pathlib import Path
import os
import logging
import plotly.express as px
import plotly.graph_objects as go

# Set paths
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #E6F0FA;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 1.1em;
        font-weight: bold;
        color: #003087;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.3em;
        color: #003087;
    }
    .section-title {
        font-size: 1.6em;
        color: #003087;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def convert_df(df):
    """Convert DataFrame to CSV format for download."""
    return df.to_csv(index=False).encode('utf-8')

def run_ml_app():
    submenu = ["Split Data", "Train Model", "Best Model", "Model by Credit Type & Segment"]
    choice = st.sidebar.selectbox("SubMenu", submenu)

    if choice == "Split Data":
        st.subheader('Diviser Votre Ensemble de Données')
        
        # Try to load the saved dataset from EDA
        df = None
        try:
            df = joblib.load(DATAPATH / 'final_dataset.pkl')
            st.session_state.df = df
            df = df[df['Impaye'] != 'unknown']
            df['Impaye'] = df['Impaye'].astype(int)
            st.write(f"Jeu de données chargé depuis EDA avec {df.shape[0]} lignes et {df.shape[1]} colonnes.")
            st.dataframe(df.head())
            logging.info(f"Loaded final_dataset.pkl with {df.shape[0]} rows.")
        except FileNotFoundError:
            st.warning("Aucun jeu de données sauvegardé trouvé dans l'onglet EDA. Veuillez charger un fichier CSV.")
            uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
            delimiter = st.selectbox("Sélectionner le délimiteur", [",", ";", "\t", " "], index=0)
            encoding = st.selectbox("Sélectionner l'encodage", ["utf-8", "latin1", "iso-8859-1", "cp1252"], index=0)
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding, on_bad_lines='warn')
                    df = df[df['Impaye'] != 'unknown']
                    df['Impaye'] = df['Impaye'].astype(int)
                    st.session_state.df = df
                    joblib.dump(df, DATAPATH / 'final_dataset.pkl')
                    st.write(f"Jeu de données chargé avec {df.shape[0]} lignes et {df.shape[1]} colonnes.")
                    st.dataframe(df.head())
                    logging.info(f"Uploaded and saved dataset with {df.shape[0]} rows to final_dataset.pkl.")
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
                    st.info("Essayez d'ajuster le délimiteur ou l'encodage.")
                    logging.error(f"Error reading CSV: {e}")
                    return
            else:
                st.info("Veuillez charger un fichier CSV ici ou dans l'onglet EDA.")
                return
        except Exception as e:
            st.error(f"Erreur lors du chargement du jeu de données : {e}")
            logging.error(f"Error loading final_dataset.pkl: {e}")
            return

        # Select target column
        target_col = st.selectbox("Sélectionner la colonne cible", ['Impaye'])
        
        # Select features
        all_features = [col for col in df.columns if col != target_col]
        num_features = ['TotalEcheance', 'capital Restant Du', 'NbreEch', 'TauxInteret', 'MontantCredit', 'AgeRelationBqe',
                        'AncienneteActivite', 'MvtConfEch', 'nb_incidents_paiement_pass_short_win',
                        'nb_incidents_paiement_pass_long_win', 'retard_moyen_pass_short_win',
                        'retard_moyen_pass_long_win', 'ratio_paiement_moyen_pass_short_win',
                        'jours_depuis_dernier_incident', 'nb_impayes_cumul']
        cat_features = ['Type de Credit_G', 'Segment_G', 'Secteur a Risque_G', 'FormeJuridique_G', 'Region']
        
        st.write("Caractéristiques numériques détectées :", num_features)
        st.write("Caractéristiques catégoriques détectées :", cat_features)
        
        # Allow user to confirm or modify features
        selected_features = st.multiselect("Sélectionner les caractéristiques à utiliser", all_features, default=num_features + cat_features)
        num_features = [f for f in selected_features if f in num_features]
        cat_features = [f for f in selected_features if f in cat_features]
        
        # Train/test split options
        st.write('Sélectionner une graine pour la reproductibilité.')
        seed = st.slider('Graine', 0, 1000, 42)
        st.write('Sélectionner la taille de l’ensemble de test en pourcentage.')
        test_size = st.slider('Test %', 10, 50, 30)
        
        if st.button('Diviser les Données en Train/Test'):
            try:
                X = df[selected_features]
                y = df[target_col].values
                df_train, df_test, X_train, X_test, y_train, y_test = train_test_split(
                    df, X, y, test_size=test_size/100, stratify=y, random_state=seed
                )
                
                # Save train/test data and features
                joblib.dump(df_train, DATAPATH / 'df_train.pkl')
                joblib.dump(df_test, DATAPATH / 'df_test.pkl')
                joblib.dump(X_train, DATAPATH / 'X_train.pkl')
                joblib.dump(X_test, DATAPATH / 'X_test.pkl')
                joblib.dump(y_train, DATAPATH / 'y_train.pkl')
                joblib.dump(y_test, DATAPATH / 'y_test.pkl')
                joblib.dump(num_features, MODELPATH / 'num_features.pkl')
                joblib.dump(cat_features, MODELPATH / 'cat_features.pkl')
                
                st.write(f'L’ensemble d’entraînement contient {df_train.shape[0]} échantillons.')
                st.dataframe(df_train[selected_features + [target_col]])
                st.write(f'L’ensemble de test contient {df_test.shape[0]} échantillons.')
                st.dataframe(df_test[selected_features + [target_col]])
                logging.info(f"Split data: train={df_train.shape[0]}, test={df_test.shape[0]}")
            except Exception as e:
                st.error(f"Erreur lors de la division des données : {e}")
                logging.error(f"Error splitting data: {e}")

    elif choice == 'Train Model':
        st.subheader('Entraîner Votre Modèle')
        if not (DATAPATH / 'X_train.pkl').exists() or not (DATAPATH / 'y_train.pkl').exists():
            st.error("Aucune donnée d’entraînement disponible. Veuillez diviser les données dans 'Split Data'.")
            logging.error("No training data for model training.")
            return
        
        st.write('Sélectionner une graine pour la reproductibilité.')
        seed = st.slider('Graine', 0, 1000, 42)

        classifiers = ["LGBM", "Logistic Regression", "XGBoost", "Random Forest", "ANN", "Decision Tree"]
        classifier_choice = st.selectbox("Classificateur", classifiers)
        
        # Load classifier
        if classifier_choice == 'LGBM':
            classifier = LGBMClassifier(random_state=seed, class_weight='balanced')
        elif classifier_choice == 'XGBoost':
            classifier = XGBClassifier(random_state=seed, scale_pos_weight=10)
        elif classifier_choice == 'Logistic Regression':
            classifier = LogisticRegression(random_state=seed, max_iter=100000, class_weight='balanced')
        elif classifier_choice == 'Random Forest':
            classifier = RandomForestClassifier(random_state=seed, class_weight='balanced')
        elif classifier_choice == 'ANN':
            classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed)
        elif classifier_choice == 'Decision Tree':
            classifier = DecisionTreeClassifier(random_state=seed, class_weight='balanced')

        # Initialize recursive feature eliminator
        rfe = RFECV(
            estimator=classifier,
            step=1,
            cv=StratifiedKFold(3),
            scoring=make_scorer(roc_auc_score),
            min_features_to_select=1
        )

        # Load features
        try:
            num_features = joblib.load(MODELPATH / 'num_features.pkl')
            cat_features = joblib.load(MODELPATH / 'cat_features.pkl')
        except FileNotFoundError:
            st.error("Fichiers de caractéristiques non trouvés. Veuillez d’abord diviser les données.")
            logging.error("num_features.pkl or cat_features.pkl not found.")
            return

        # Setup pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('onehotencode', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ],
            remainder='drop'
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('rfe', rfe),
            ('classifier', classifier)
        ])

        # Fit the model
        if st.button('Entraîner le Modèle'):
            with st.spinner('Entraînement du modèle...'):
                try:
                    X_train = joblib.load(DATAPATH / 'X_train.pkl')
                    y_train = joblib.load(DATAPATH / 'y_train.pkl')
                    pipeline.fit(X_train, y_train)
                    joblib.dump(pipeline, MODELPATH / 'pipeline.pkl')
                    st.success('Modèle entraîné et sauvegardé avec succès !')
                    logging.info(f"Trained {classifier_choice} model and saved pipeline.pkl.")
                except Exception as e:
                    st.error(f"Erreur lors de l’entraînement du modèle : {e}")
                    logging.error(f"Error training model: {e}")

        # Predict on test set
        if st.button('Prédire le Risque de Défaut pour l’Ensemble de Test'):
            try:
                df_test = joblib.load(DATAPATH / 'df_test.pkl')
                X_test = joblib.load(DATAPATH / 'X_test.pkl')
                pipeline = joblib.load(MODELPATH / 'pipeline.pkl')
                
                y_scores = pipeline.predict_proba(X_test)[:, 1]
                df_test['predicted_credit_default_scores'] = y_scores
                
                st.write("Ensemble de test avec scores de risque de défaut prédits :")
                st.dataframe(df_test)
                
                # Download predictions
                csv = convert_df(df_test)
                st.download_button(
                    label="Télécharger les données sous forme de CSV",
                    data=csv,
                    file_name='data_scored.csv',
                    mime='text/csv'
                )
                logging.info("Generated predictions for test dataset.")
            except FileNotFoundError:
                st.error("Données de test ou modèle non trouvés. Veuillez d’abord diviser les données et entraîner un modèle.")
                logging.error("df_test.pkl, X_test.pkl, or pipeline.pkl not found.")
            except Exception as e:
                st.error(f"Erreur lors des prédictions : {e}")
                logging.error(f"Error making predictions: {e}")

    elif choice == 'Best Model':
        st.subheader('Meilleur Modèle')
        if not (DATAPATH / 'X_train.pkl').exists() or not (DATAPATH / 'y_train.pkl').exists():
            st.error("Aucune donnée d’entraînement disponible. Veuillez diviser les données dans 'Split Data'.")
            return

        # Load data and features
        try:
            X_train = joblib.load(DATAPATH / 'X_train.pkl')
            y_train = joblib.load(DATAPATH / 'y_train.pkl')
            X_test = joblib.load(DATAPATH / 'X_test.pkl')
            y_test = joblib.load(DATAPATH / 'y_test.pkl')
            num_features = joblib.load(MODELPATH / 'num_features.pkl')
            cat_features = joblib.load(MODELPATH / 'cat_features.pkl')
        except FileNotFoundError:
            st.error("Fichiers de données ou de caractéristiques non trouvés. Veuillez diviser les données.")
            return

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'XGBoost': XGBClassifier(scale_pos_weight=10, random_state=42),
            'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42),
            'ANN': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }

        # Setup preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehotencode', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ]
        )

        # Train and evaluate models
        if st.button('Évaluer Tous les Modèles'):
            with st.spinner('Entraînement et évaluation des modèles...'):
                results = []
                pipelines = {}
                for name, model in models.items():
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])
                    pipelines[name] = pipeline
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
                    
                    results.append({
                        'Modèle': name,
                        'Recall': recall_score(y_test, y_pred, pos_label=1),
                        'ROC-AUC': roc_auc_score(y_test, y_pred_prob),
                        'F1-Score': f1_score(y_test, y_pred, pos_label=1),
                        'Precision': precision_score(y_test, y_pred, pos_label=1)
                    })

                # Display results
                results_df = pd.DataFrame(results)
                best_model = results_df.loc[results_df['Recall'].idxmax(), 'Modèle']
                st.markdown('<p class="section-title">📊 Résumé des Performances des Modèles</p>', unsafe_allow_html=True)
                st.dataframe(results_df.style.highlight_max(subset=['Recall'], color='#E6F0FA'))
                st.write(f"**Meilleur modèle basé sur le Recall : {best_model}**")
                
                # Save best model
                joblib.dump(pipelines[best_model], MODELPATH / 'best_pipeline.pkl')
                logging.info(f"Saved best model {best_model} to best_pipeline.pkl")

                # Feature importance for compatible models
                st.markdown('<p class="section-title">📈 Importance des Caractéristiques</p>', unsafe_allow_html=True)
                for name, pipeline in pipelines.items():
                    if name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM']:
                        importance = pipeline.named_steps['classifier'].feature_importances_ if name != 'Logistic Regression' else abs(pipeline.named_steps['classifier'].coef_[0])
                        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                        feature_importance = pd.DataFrame({'Caractéristique': feature_names, 'Importance': importance}).sort_values('Importance', ascending=False).head(10)
                        
                        fig = px.bar(feature_importance, x='Importance', y='Caractéristique', title=f'Top 10 Importance des Caractéristiques - {name}',
                                     color_discrete_sequence=['#003087'])
                        st.plotly_chart(fig, use_container_width=True)

        # Predict with best model
        if st.button('Prédire avec le Meilleur Modèle'):
            try:
                df_test = joblib.load(DATAPATH / 'df_test.pkl')
                X_test = joblib.load(DATAPATH / 'X_test.pkl')
                pipeline = joblib.load(MODELPATH / 'best_pipeline.pkl')
                
                y_scores = pipeline.predict_proba(X_test)[:, 1]
                df_test['predicted_credit_default_scores'] = y_scores
                
                st.write("Ensemble de test avec scores de risque prédits (Meilleur Modèle) :")
                st.dataframe(df_test)
                
                csv = convert_df(df_test)
                st.download_button(
                    label="Télécharger les prédictions sous forme de CSV",
                    data=csv,
                    file_name='best_model_predictions.csv',
                    mime='text/csv'
                )
                logging.info("Generated predictions with best model.")
            except FileNotFoundError:
                st.error("Données de test ou meilleur modèle non trouvés. Veuillez évaluer les modèles d’abord.")
            except Exception as e:
                st.error(f"Erreur lors des prédictions : {e}")

    elif choice == 'Model by Credit Type & Segment':
        st.subheader('Modélisation par Type de Crédit et Segment de Clientèle')
        if not (DATAPATH / 'X_train.pkl').exists() or not (DATAPATH / 'y_train.pkl').exists():
            st.error("Aucune donnée d’entraînement disponible. Veuillez diviser les données dans 'Split Data'.")
            return

        # Load data and features
        try:
            df_train = joblib.load(DATAPATH / 'df_train.pkl')
            X_train = joblib.load(DATAPATH / 'X_train.pkl')
            y_train = joblib.load(DATAPATH / 'y_train.pkl')
            num_features = joblib.load(MODELPATH / 'num_features.pkl')
            cat_features = joblib.load(MODELPATH / 'cat_features.pkl')
        except FileNotFoundError:
            st.error("Fichiers de données ou de caractéristiques non trouvés. Veuillez diviser les données.")
            return

        # Define preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]), num_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencode', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True))
                ]), [c for c in cat_features if c not in ['Type de Credit_G', 'Segment_G']])
            ]
        )

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42),
            'ANN': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }

        def train_and_evaluate_subset(df_subset, subset_name, group_type, seed=42, test_size=0.3):
            if len(df_subset) < 10:
                st.warning(f"{group_type} {subset_name} : Données insuffisantes ({len(df_subset)} échantillons).")
                return None, None
            if len(df_subset['Impaye'].unique()) < 2:
                st.warning(f"{group_type} {subset_name} : Une seule classe présente.")
                return None, None

            X = df_subset[num_features + [c for c in cat_features if c not in ['Type de Credit_G', 'Segment_G']]]
            y = df_subset['Impaye']
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
            
            results = []
            pipelines = {}
            for name, model in models.items():
                if name == 'XGBoost':
                    scale_pos_weight = len(y_train_sub[y_train_sub == 0]) / len(y_train_sub[y_train_sub == 1]) if len(y_train_sub[y_train_sub == 1]) > 0 else 1
                    model.set_params(scale_pos_weight=scale_pos_weight)
                
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                pipeline.fit(X_train_sub, y_train_sub)
                pipelines[name] = pipeline
                
                y_pred = pipeline.predict(X_test_sub)
                y_pred_prob = pipeline.predict_proba(X_test_sub)[:, 1]
                
                results.append({
                    'Modèle': name,
                    'Recall': recall_score(y_test_sub, y_pred, pos_label=1),
                    'ROC-AUC': roc_auc_score(y_test_sub, y_pred_prob),
                    'F1-Score': f1_score(y_test_sub, y_pred, pos_label=1),
                    'Precision': precision_score(y_test_sub, y_pred, pos_label=1)
                })

            results_df = pd.DataFrame(results)
            best_model = results_df.loc[results_df['Recall'].idxmax(), 'Modèle']
            return results_df, pipelines[best_model]
#
        if st.button('Entraîner Modèles par Type de Crédit et Segment'):
            with st.spinner('Entraînement des modèles segmentés...'):
                best_pipelines = {}
                
                # By Credit Type
                st.markdown('<p class="section-title">📊 Résultats par Type de Crédit</p>', unsafe_allow_html=True)
                for credit_type in df_train['Type de Credit_G'].unique():
                    df_subset = df_train[df_train['Type de Credit_G'] == credit_type]
                    results_df, best_pipeline = train_and_evaluate_subset(df_subset, credit_type, "Type de Crédit")
                    if results_df is not None:
                        st.write(f"**Type de Crédit : {credit_type}**")
                        st.dataframe(results_df.style.highlight_max(subset=['Recall'], color='#E6F0FA'))
                        st.write(f"Meilleur modèle : {results_df.iloc[results_df['Recall'].idxmax()]['Modèle']}")
                        best_pipelines[f'type_credit_{credit_type}'] = best_pipeline
                # Save best pipelines
                joblib.dump(best_pipelines, MODELPATH / 'segmented_pipelines.pkl')
                logging.info("Saved segmented pipelines.")


                # By Segment
                st.markdown('<p class="section-title">📅 Résultats par Segment de Clientèle</p>', unsafe_allow_html=True)
                for segment in df_train['Segment_G'].unique():
                    df_subset = df_train[df_train['Segment_G'] == segment]
                    results_df, best_pipeline = train_and_evaluate_subset(df_subset, segment, "Segment_G")
                    if results_df is not None:
                        st.write(f"**Segment : {segmentation}**")
                        st.dataframe(results_df.style.highlight_max(subset=['Recall'], color='#E6F0FA'))
                        st.write(f"Meilleur modèle : {results_df.iloc[results_df['Recall'].idxmax()]['Modèle']}")
                        best_pipelines[f'segment_{segment}'] = best_pipeline

                # Save best pipelines
                joblib.dump(best_pipelines, MODELPATH / 'segmented_pipelines.pkl')
                logging.info("Saved segmented pipelines.")

        # Predict with segmented models
        if st.button('Prédire avec les Modèles Segmentés'):
            try:
                df_test = joblib.load(DATAPATH / 'df_test.pkl')
                best_pipelines = joblib.load(MODELPATH / 'segmented_pipelines.pkl')
                
                df_test['Impayé_Probabilité'] = np.nan
                for key, pipeline in best_pipelines.items():
                    if key.startswith('type_credit_'):
                        credit_type = key.replace('type_credit_', '')
                        mask = df_test['Type de Credit_G'] == credit_type
                        if mask.any():
                            X = df_test[mask][num_features + [c for c in cat_features if c not in ['Type de Credit_G', 'Segment_G']]]
                            df_test.loc[mask, 'Impayé_Probabilité'] = pipeline.predict_proba(X)[:, 1]
                    elif key.startswith('segment_'):
                        segment = key.replace('segment_', '')
                        mask = (df_test['Segment_G'] == segment) & (df_test['Impayé_Probabilité'].isna())
                        if mask.any():
                            X = df_test[mask][num_features + [c for c in cat_features if c not in ['Type de Credit_G', 'Segment_G']]]
                            df_test.loc[mask, 'Impayé_Probabilité'] = pipeline.predict_proba(X)[:, 1]

                st.write("Ensemble de test avec probabilités d’impayés segmentées :")
                st.dataframe(df_test)
                
                csv = convert_df(df_test)
                st.download_button(
                    label="Télécharger les prédictions segmentées",
                    data=csv,
                    file_name='segmented_predictions.csv',
                    mime='text/csv'
                )
                logging.info("Generated segmented predictions.")
            except Exception as e:
                st.error(f"Erreur lors des prédictions segmentées : {e}")




if __name__ == "__main__":
    run_ml_app()