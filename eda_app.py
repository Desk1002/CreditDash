import streamlit as st
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime 
from utils import plot_num_feature_distribution, plot_cat_feature_distribution, plot_target_distribution

rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)

#*************************************************************************

def load(train, date_arrete_donnees, seuil_retard_critique_jours, seuil_paiement_incomplet_ratio, window_behavior_short, window_behavior_long):
    '''
    Preprocess the dataset and calculate behavioral features.
    
    Parameters:
    train (pandas.DataFrame): Input DataFrame.
    date_arrete_donnees (datetime): Cut-off date for data processing.
    seuil_retard_critique_jours (int): Threshold for critical overdue days.
    seuil_paiement_incomplet_ratio (float): Threshold for incomplete payment ratio.
    window_behavior_short (int): Short window for behavioral indicators.
    window_behavior_long (int): Long window for behavioral indicators.
    
    Returns:
    pandas.DataFrame: Processed DataFrame with behavioral features.
    '''
    train=train.drop_duplicates()

    import unicodedata
    columnsNameNomalizer = lambda x: str(
        unicodedata.normalize('NFKD', x).encode('ascii', 'ignore'
                        ).decode('utf-8')).title().replace('_', '').strip() if '_' in str(x) else str(
        unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')).strip()

    train.rename(columns={c: columnsNameNomalizer(c) for c in train.columns}, inplace=True)

    train['FormeJuridique']=train['Forme Juridique'].str.split(' ', n=0).str[0]

    # --- PARAMÈTRES CRITIQUES ---
    DATE_ARRETE_DONNEES = pd.to_datetime(date_arrete_donnees) # Exemple : "2025-04-20"
    SEUIL_RETARD_CRITIQUE_JOURS = seuil_retard_critique_jours # Exemple : 0, 30 jours de retard
    SEUIL_PAIEMENT_INCOMPLET_RATIO =  seuil_paiement_incomplet_ratio  # Exemple : paiement < 95% du total dû
    WINDOW_BEHAVIOR_SHORT = window_behavior_short # Fenêtre courte pour indicateurs comportementaux (ex: 3 dernières échéances)
    WINDOW_BEHAVIOR_LONG = window_behavior_long  # Fenêtre longue pour indicateurs comportementaux (ex: 6 dernières échéances)

    df_behavior = train.copy()

    # Supprimer la colonne 'Unnamed: 0' si elle existe et est un simple index
    if 'Unnamed: 0' in df_behavior.columns:
        df_behavior = df_behavior.drop('Unnamed: 0', axis=1)

    # Conversion des colonnes de dates essentielles
    date_cols = ["Date d'Echeance", "Date comptable Hist", "DateMisePlace"]
    for col in date_cols:
        df_behavior[col] = pd.to_datetime(df_behavior[col], errors='coerce')

    # Tri des données : crucial pour les calculs de lag et rolling par client/dossier
    df_behavior = df_behavior.sort_values(by=['Code Client', 'N du Dossier', "Date d'Echeance", 'N Echeance'])


    # --- PARTIE 1: CALCUL DES INDICATEURS DE BASE POUR LE COMPORTEMENT PASSÉ ---
    # Ces indicateurs sont calculés pour chaque échéance historique où l'information de paiement est disponible.
    # Ils serviront de base pour créer les features comportementales agrégées.

    df_behavior['retard_jours_effectif'] = np.nan
    df_behavior['ratio_paiement_effectif'] = np.nan
    df_behavior['paiement_effectif_partiel_ou_retard'] = 0 # 0: non, 1: oui

    mask_paiement_info_dispo = df_behavior["Date comptable Hist"].notna() & df_behavior["Montant Mvt Hist"].notna()

    # Calcul du retard effectif pour les paiements enregistrés
    df_behavior.loc[mask_paiement_info_dispo, 'retard_jours_effectif'] = \
        (df_behavior.loc[mask_paiement_info_dispo, "Date comptable Hist"] - df_behavior.loc[mask_paiement_info_dispo, "Date d'Echeance"]).dt.days

    # Calcul du ratio de paiement effectif
    df_behavior.loc[mask_paiement_info_dispo, 'ratio_paiement_effectif'] = \
        df_behavior.loc[mask_paiement_info_dispo, "Montant Mvt Hist"] / (df_behavior.loc[mask_paiement_info_dispo, "TotalEcheance"] + 1e-6) # Eviter division par zero

    # Marquer si un paiement effectif a été en retard critique OU partiel critique
    condition_retard = df_behavior['retard_jours_effectif'] > SEUIL_RETARD_CRITIQUE_JOURS
    condition_partiel = df_behavior['ratio_paiement_effectif'] < SEUIL_PAIEMENT_INCOMPLET_RATIO
    df_behavior.loc[mask_paiement_info_dispo & (condition_retard | condition_partiel), 'paiement_effectif_partiel_ou_retard'] = 1

    df_target = df_behavior.copy() # Utiliser le df déjà trié et avec dates converties

    df_target['Impaye'] = 0 # Initialisation

    for index, row in df_target.iterrows():
        DE = row["Date d'Echeance"]
        ME = row["TotalEcheance"]
        DP = row["Date comptable Hist"] # Déjà en datetime (ou NaT)
        MP = row["Montant Mvt Hist"]

        # On ne crée la cible que pour les échéances dont la date est <= DATE_ARRETE_DONNEES
        if DE > DATE_ARRETE_DONNEES:
            #df_target.loc[index, 'Impaye = np.nan # Statut inconnu pour le futur
            df_target.loc[index, 'Impaye'] = 'unknown'  # Statut inconnu pour le futur
            continue

        # Règle 1: Données de paiement disponibles
        if pd.notna(DP) and pd.notna(MP):
            retard_jours = (DP - DE).days # DP et DE sont des datetime
            ratio_paiement = MP / ME if ME > 0 else 0

            if retard_jours > SEUIL_RETARD_CRITIQUE_JOURS:
                df_target.loc[index, 'Impaye']=1
            elif ratio_paiement < SEUIL_PAIEMENT_INCOMPLET_RATIO:
                df_target.loc[index, 'Impaye']=1
            # else: Y reste 0

        # Règle 2: Données de paiement manquantes pour une échéance passée
        elif pd.isna(DP) or pd.isna(MP):
            # Si l'échéance est suffisamment passée par rapport à la date d'arrêté
            # et que le délai critique pour le paiement est dépassé, on considère comme impayé.
            if DE <= (DATE_ARRETE_DONNEES - pd.Timedelta(days=SEUIL_RETARD_CRITIQUE_JOURS)):
                df_target.loc[index, 'Impaye']=1
            else:
                # Échéance passée mais délai critique non encore atteint depuis DE jusqu'à DATE_ARRETE_DONNEES
                # pour conclure un impayé sur la base des données manquantes.
                # Ces cas sont ambigus pour l'entraînement si on ne peut pas affirmer un non-paiement.
                df_target.loc[index, 'Impaye'] = 'unknown' #np.nan

    df_final = df_target.copy()

    # S'assurer que le DataFrame est trié correctement
    df_final = df_final.sort_values(by=['Code Client', 'N du Dossier', "Date d'Echeance", 'N Echeance'])

    # Grouper par 'N du Dossier' (ou 'Code Client' si plus pertinent pour le comportement)
    # Utiliser 'N du Dossier' semble plus spécifique à un prêt.
    # Le .shift(1) est crucial pour n'utiliser que les données des échéances précédentes.

    # 1. Nombre d'impayés (paiements en retard ou partiels) passés sur N fenêtres glissantes
    #    Utilise la colonne 'paiement_effectif_partiel_ou_retard' calculée précédemment.
    #    On somme sur une fenêtre glissante des N dernières échéances (excluant l'actuelle).
    df_final[f'nb_incidents_paiement_pass_short_win'] = df_final.groupby('N du Dossier')['paiement_effectif_partiel_ou_retard'].transform(
        lambda x: x.shift(1).rolling(window=WINDOW_BEHAVIOR_SHORT, min_periods=1).sum()
    )
    df_final[f'nb_incidents_paiement_pass_long_win'] = df_final.groupby('N du Dossier')['paiement_effectif_partiel_ou_retard'].transform(
        lambda x: x.shift(1).rolling(window=WINDOW_BEHAVIOR_LONG, min_periods=1).sum()
    )

    # 2. Retard moyen sur les N dernières échéances payées
    df_final[f'retard_moyen_pass_short_win'] = df_final.groupby('N du Dossier')['retard_jours_effectif'].transform(
        lambda x: x.shift(1).rolling(window=WINDOW_BEHAVIOR_SHORT, min_periods=1).mean()
    )
    df_final[f'retard_moyen_pass_long_win'] = df_final.groupby('N du Dossier')['retard_jours_effectif'].transform(
        lambda x: x.shift(1).rolling(window=WINDOW_BEHAVIOR_LONG, min_periods=1).mean()
    )

    # 3. Ratio de paiement moyen sur les N dernières échéances payées
    df_final[f'ratio_paiement_moyen_pass_short_win'] = df_final.groupby('N du Dossier')['ratio_paiement_effectif'].transform(
        lambda x: x.shift(1).rolling(window=WINDOW_BEHAVIOR_SHORT, min_periods=1).mean()
    )

    # 4. Jours depuis le dernier incident de paiement (retard ou partiel)
    #    Nécessite de marquer les dates des incidents.
    df_final['date_incident_paiement'] = df_final.loc[df_final['paiement_effectif_partiel_ou_retard'] == 1, "Date d'Echeance"]
    df_final['date_dernier_incident_passe'] = df_final.groupby('N du Dossier')['date_incident_paiement'].transform(
        lambda x: x.shift(1).ffill() # Propage la dernière date d'incident connue
    )
    df_final['jours_depuis_dernier_incident'] = (df_final["Date d'Echeance"] - df_final['date_dernier_incident_passe']).dt.days
    df_final.drop(columns=['date_incident_paiement', 'date_dernier_incident_passe'], inplace=True) # Nettoyage colonnes temporaires

    # 5. Indicateur de régularité : nombre d'échéances consécutives payées sans incident avant l'actuelle
    #    On crée un compteur qui se réinitialise à chaque incident.
    df_final['sans_incident_actuel'] = 1 - df_final['paiement_effectif_partiel_ou_retard'] # 1 si pas d'incident, 0 si incident
    # Pour chaque dossier, on calcule les cumuls consécutifs de 'sans_incident_actuel'
    # Le cumsum est groupé par les incidents.
    # Puis on prend le shift(1) de ce compte pour avoir le nombre *avant* l'échéance actuelle.
    # Cette logique est un peu plus complexe pour le cumul consécutif, voici une approche simplifiée pour l'exemple:
    # On peut utiliser une logique de comptage depuis le dernier incident.
    # Si 'jours_depuis_dernier_incident' est grand et positif, c'est une bonne indication.

    #***********************************************************************************************************************************
    # Cela est crucial pour que .shift() fonctionne comme attendu (regarder le passé)
    df_final = df_final.sort_values(by=['Code Client', 'N du Dossier', "Date d'Echeance", 'N Echeance'])
    # Conversion sécurisée si nécessaire (au cas où Target_Y serait de type object)
    df_final['Impaye_clean'] = df_final['Impaye'].replace('unknown', np.nan).astype(float)

    #6. Calcul du nombre d'impayés avant l'échéance actuelle 

    # On utilise Target_Y des échéances précédentes.
    # Target_Y = 1 signifie un impayé. Target_Y = 0 ou NaN signifie non-impayé pour ce comptage.
    # Le .shift(1) décale les valeurs d'une ligne vers le bas au sein de chaque groupe,
    # ainsi pour l'échéance actuelle, on regarde la valeur de Target_Y de l'échéance précédente.

    # Option 1: Nombre total d'impayés passés pour ce dossier
    # .cumsum() fait la somme cumulative. .shift(1) utilise la somme cumulative *avant* l'échéance actuelle.
    df_final['nb_impayes_cumul'] = df_final.groupby('N du Dossier')['Impaye_clean'].transform(
        lambda x: x.fillna(0).shift(1).cumsum() # Remplace NaN par 0 pour la somme, puis décale et cumule
    )
    # Remplir les NaN pour la première échéance de chaque dossier (résultant du shift) par 0
    df_final['nb_impayes_cumul'].fillna(0, inplace=True)
    #******************************************************************

    # 7. Ratio d'impayés cumulés sur le total des échéances passées
    # On peut aussi calculer le ratio d'impayés cumulés sur le total des échéances passées. 
    df_final['ratio_impayes']= df_final['nb_impayes_cumul'] / df_final.groupby('N du Dossier')['Impaye_clean'].transform(
        lambda x: x.fillna(0).shift(1).count() # Compte le nombre d'échéances passées
    )


    # Remplacer les NaN initiaux des features comportementales (dus au shift/rolling sur les premières échéances) par 0 ou une autre valeur
    cols_behavior_features = [
        f'nb_incidents_paiement_pass_short_win', f'nb_incidents_paiement_pass_long_win',
        f'retard_moyen_pass_short_win', f'retard_moyen_pass_long_win',
        f'ratio_paiement_moyen_pass_short_win', 'jours_depuis_dernier_incident'
    ]
    for col in cols_behavior_features:
        # Pour les retards moyens et ratios, NaN peut signifier "aucun paiement antérieur" ou "aucun paiement avec info".
        # Pour les jours_depuis_dernier_incident, NaT/NaN signifie "aucun incident antérieur".
        # On pourrait imputer avec une valeur qui a du sens (ex: 0 pour nb_incidents, une grande valeur pour jours_depuis_dernier_incident si aucun incident)
        if 'nb_incidents' in col:
            df_final[col].fillna(0, inplace=True)
        elif 'retard_moyen' in col or 'ratio_paiement_moyen' in col:
            df_final[col].fillna(df_final[col].median(), inplace=True) # Ou 0, ou une autre imputation logique
        elif 'jours_depuis_dernier_incident' in col:
            df_final[col].fillna(9999, inplace=True) # Grande valeur si aucun incident passé

    df_final.drop(['Impaye_clean', 'Forme Juridique'],axis=1, inplace=True) 

    df_final1=df_final.drop(['Gestionnaire', 'Nom Agence', 'Nom Client'], axis=1) 
    # Print the categories for each categorical attribute
    pd.set_option('display.max_colwidth', 0)
    cat = []
    for col_name in df_final1.select_dtypes(include=['object']).columns:
        categories_list = df_final1[col_name].value_counts().index.to_list()
        cat.append([col_name, categories_list])
    pd.DataFrame(cat, columns=["Column Name", "Classes"]).set_index("Column Name").rename_axis(None)

    # Identification des types de colonnes  
    numerical_cols = df_final.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_final.select_dtypes(include='object').columns.tolist()

    # Exclure les identifiants et la cible des colonnes numériques à traiter 
    ids_and_target = ['N Echeance', 'Code Client','N compte', 'N du Dossier','Code Agence', 'Code Gestionnaire', 'Impaye', ] # N° compte est float mais plus un ID
    numerical_features = [col for col in numerical_cols if col not in ids_and_target]

    cols_to_encode = ['Type de Credit', 'Segment', 'Secteur a Risque', 'FormeJuridique'] #'Nom Agence'
    # Exclure 'Nom Client' et autres identifiants textuels pour l'encodage direct
    categorical_features_to_encode = [col for col in cols_to_encode if col in df_final.columns]

    # Fonction pour nettoyer les espaces et accents dans Nom Agence
    def clean_string(text):
        if pd.isna(text):  # Gérer les valeurs manquantes
            return text
        # Remplacer les espaces par des underscores
        text = text.replace(' ', '_')
        return text

    # Appliquer le nettoyage à la colonne 'Nom Agence'
    df_final['Nom Agence'] = df_final['Nom Agence'].apply(clean_string)

    # Définir le mapping des localités vers les régions
    location_to_region = {
        # Centre (Yaoundé et environs)
        'HIPPODROME': 'Centre', 'MENDONG': 'Centre', 'RETRAITE': 'Centre', 'MVOG-MBI': 'Centre',
        'MESSA': 'Centre', 'BIYEM-ASSI': 'Centre', 'ESSOS': 'Centre', 'ETOUDI': 'Centre',
        'OMNISPORT': 'Centre', 'MFOUNDI': 'Centre', 'BASTOS': 'Centre', 'SAINT-MICHEL': 'Centre',
        'MIMBOMAN': 'Centre', 'MELEN': 'Centre', 'DAMAS': 'Centre', 'MARCHE_CENTRAL': 'Centre',
        'NKOABANG': 'Centre', 'NKOLBISSON': 'Centre', 'MVAN': 'Centre', 'AHALA': 'Centre',
        'NKOLBONG': 'Centre', 'MFOU': 'Centre', 'YADEME': 'Centre', 'OLEMBE': 'Centre', 'CAMAIR': 'Centre',
        
        # Littoral (Douala et environs)
        'BONANJO': 'Littoral', 'BONABERI': 'Littoral', 'AKWA': 'Littoral', 'BESSENGUE': 'Littoral',
        'NEW-BELL': 'Littoral', 'MBOPPI': 'Littoral', 'BONAMOUSSADI': 'Littoral', 'NDOKOTTI': 'Littoral',
        'AKWA_MILLENIUM': 'Littoral', 'BONABERI_II': 'Littoral', 'MESSAMENDONGO': 'Littoral',
        'CITE_DES_PALMIERS': 'Littoral', 'YASSA': 'Littoral', 'PORT_AUTONOME_DOUALA': 'Littoral',
        'KOUMASSI': 'Littoral', 'MAKEPE': 'Littoral', 'BONAPRISO': 'Littoral', 'NDOG-PASSI': 'Littoral',
        'DOUCHE_MUNICIPALE': 'Littoral', 'NKONGSAMBA': 'Littoral', 'EDEA': 'Littoral', 'LOUM': 'Littoral',
        
        # West
        'BAFOUSSAM': 'West', 'MBOUDA': 'West', 'BALENG': 'West', 'DSCHANG': 'West', 'FOUMBOT': 'West',
        
        # North
        'GAROUA': 'North', 'MEIGANGA': 'North', 'TAMDJA': 'North',
        
        # Far North
        'MAROUA': 'Far North', 'KOUSSERI': 'Far North', 'MOKOLO': 'Far North',
        
        # East
        'BERTOUA': 'East',
        
        # South
        'KRIBI': 'South', 'EBOLOWA': 'South', 'SANGMELIMA': 'South',
        
        # Southwest
        'LIMBE': 'Southwest', 'KUMBA': 'Southwest', 'BUEA': 'Southwest',
        
        # Northwest
        'BAMENDA': 'Northwest',
        
        # Adamawa
        'NGAOUNDERE': 'Adamawa',
        
        # Cas particulier
        'DAKAR': 'Littoral',  
        'CTX_CENTRE-SUD-EST-': 'Centre'  # Contexte suggère une localisation dans le Centre-Sud-Est
    }

    # Fonction pour extraire la localité et mapper à la région
    def map_to_region(branch_name):
        if pd.isna(branch_name):
            return None
        location = branch_name.replace('FIRST_BANK_', '')
        return location_to_region.get(location, 'Unknown')  # 'Unknown' si non mappé

    # Appliquer la catégorisation à la colonne 'Nom Agence'
    df_final['Region'] = df_final['Nom Agence'].apply(map_to_region)

    
    
    # Définir les mappings pour le regroupement des classes
    segment_mapping = {
        'PE': 'TPE_PE','ME': 'ME','GE': 'GE','TPE': 'TPE_PE',
        'INS': 'Autre', 'ASS': 'Autre', 'PAR': 'Autre'
    }

    secteur_risque_mapping = {
        'Commerce': 'Commerce', 'Production Autres Services': 'Production Autres Services',
        'Transports': 'Transport','Industries': 'Industries', 'Construction': 'Construction',
        'Energie': 'Autres', 'Activités Financières et Assurance': 'Autres','Activités Agro-Pastorales': 'Autres',
        'Télécommunications': 'Autres', 'Secteur Public': 'Autres', 'Autres Secteurs': 'Autres'
    }

    type_credit_mapping = {
        'Credit court terme tresorerie': 'Credit court terme tresorerie',
        'Credit moyen terme equipement': 'Credit moyen terme equipement',
        'Credit court terme mar.publics': 'Autres',
        'Credit court terme escpte doc': 'Autres', 'CMT investissement immobilier': 'Autres',
        'CMT non ventilable': 'Autres', 'Credit court terme equipement': 'Autres',
        'Credit court terme Campagne': 'Autres', 'CMT campagne': 'Autres', 'Credit court terme non ventil.': 'Autres',
        'CLT non ventilable': 'Autres','CMT marches publics': 'Autres', 'CMT habitat': 'Autres',
        'Credit court terme morat. etat': 'Autres'
    }

    forme_juridique_mapping = {
        'SARL': 'SARL','EURL': 'EURL', 'SA': 'SA', 'SCI': 'Autres', 'ASDP': 'Autres','SASU': 'Autres','SAS': 'Autres',
        'ASCI': 'Autres','SCOP': 'Autres','ASCO': 'Autres', 'GIE': 'Autres','ASDPC': 'Autres','SM': 'Autres','ASC': 'Autres',
        'SCA': 'Autres','SCEC': 'Autres','SP': 'Autres','OP': 'Autres','SCP': 'Autres','SCIC': 'Autres'
    }

    # Appliquer les mappings au DataFrame
    df_final['Segment_G'] = df_final['Segment'].map(segment_mapping).fillna('Autre')
    df_final['Type de Credit_G'] = df_final['Type de Credit'].map(type_credit_mapping).fillna('Autres')
    df_final['FormeJuridique_G'] = df_final['FormeJuridique'].map(forme_juridique_mapping).fillna('Autres')
    df_final['Secteur a Risque_G'] = df_final['Secteur a Risque'].map(secteur_risque_mapping).fillna('Autres')
    return df_final

#*************************************************************************************

def categorize_variables(df):
    """
    Categorize numerical variables based on quartiles Q1, Q2, Q3.
    Adds new columns with '_Cat' suffix containing the categories.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame containing the numerical variables.
    
    Returns:
    pandas.DataFrame: DataFrame with additional categorized columns.
    """
    df = df.copy()
    
    # Define quartile thresholds for each variable
    thresholds = {
        'N Echeance': {'Q1': 1, 'Q2': 5, 'Q3': 12},
        'TotalEcheance': {'Q1': 913430.27, 'Q2': 3579784.65, 'Q3': 13686171.13},
        'capital Restant Du': {'Q1': 166853.9125, 'Q2': 5232184.46, 'Q3': 58029055.73},
        'NbreEch': {'Q1': 1, 'Q2': 12, 'Q3': 36},
        'TauxInteret': {'Q1': 9, 'Q2': 14.5, 'Q3': 14.5},
        'MontantCredit': {'Q1': 6740000, 'Q2': 30000000, 'Q3': 150000000},
        'N compte': {'Q1': 3513181001, 'Q2': 5165111001, 'Q3': 6617746001},
        'Montant Mvt Hist': {'Q1': 774407, 'Q2': 3624356, 'Q3': 14157930},
        'AgeRelationBqe': {'Q1': 4, 'Q2': 8, 'Q3': 12},
        'AncienneteActivite': {'Q1': 6, 'Q2': 10, 'Q3': 17},
        'ChiffreAffaire': {'Q1': 25000000, 'Q2': 200000000, 'Q3': 1673456000},
        'MvtConfEch': {'Q1': 600000, 'Q2': 11886055.5, 'Q3': 92103660},
        'SoldeMoyEch': {'Q1': 2862.5, 'Q2': 1119991.5, 'Q3': 11533964},
        'Code Agence': {'Q1': 2, 'Q2': 8, 'Q3': 29},
        'retard_jours_effectif': {'Q1': -154, 'Q2': 0, 'Q3': 2},
        'nb_incidents_paiement_pass_short_win': {'Q1': 0, 'Q2': 1, 'Q3': 2},
        'nb_incidents_paiement_pass_long_win': {'Q1': 0, 'Q2': 1, 'Q3': 3},
        'jours_depuis_dernier_incident': {'Q1': 31, 'Q2': 335, 'Q3': 9999},
        'nb_impayes_cumul': {'Q1': 0, 'Q2': 1, 'Q3': 4},
    }

    # Categorization function
    def categorize_value(x, q1, q2, q3):
        if pd.isna(x):
            return 'Missing'
        if x <= q1:
            return 'Faible'
        elif x <= q2:
            return 'Moyen-Bas'
        elif x <= q3:
            return 'Moyen-Haut'
        else:
            return 'Élevé'

    # Special cases for binary variables
    def categorize_binary(x):
        if pd.isna(x):
            return 'Missing'
        return 'Oui' if x == 1 else 'Non'

    # Special cases for variables with Q2=Q3 or concentrated distributions
    def categorize_concentrated(x, q1, q2, q3):
        if pd.isna(x):
            return 'Missing'
        if x <= q1:
            return 'Faible'
        elif x <= q2:
            return 'Moyen'
        else:
            return 'Élevé'

    # Apply categorizations
    for col in thresholds.keys():
        if col in df.columns:
            if col in ['paiement_effectif_partiel_ou_retard', 'sans_incident_actuel']:
                df[f'{col}_Cat'] = df[col].apply(categorize_binary)
            elif col in ['TauxInteret', 'retard_moyen_pass_short_win', 'sans_incident_actuel']:
                df[f'{col}_Cat'] = df[col].apply(
                    lambda x: categorize_concentrated(x, thresholds[col]['Q1'], thresholds[col]['Q2'], thresholds[col]['Q3'])
                )
            else:
                df[f'{col}_Cat'] = df[col].apply(
                    lambda x: categorize_value(x, thresholds[col]['Q1'], thresholds[col]['Q2'], thresholds[col]['Q3'])
                )

    return df

##################################################################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import os
import logging
import numpy as np

# Set paths
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)

def preprocess_dataset(df):
    """Preprocess dataset to ensure consistent types for categorical columns."""
    for col in df.columns:
        # Identify potential categorical columns (object or low-cardinality numeric)
        if df[col].dtype == 'object' or (df[col].dtype in ['int64', 'float64'] and df[col].nunique() < 20):
            # Convert to string, filling NaN with 'Unknown'
            df[col] = df[col].astype(str).fillna(' ')
    return df

def run_eda_app():
    submenu = ["Upload Data File", "Descriptive Stats", "Data Types", "Target Distribution", "Feature Distribution", "Credit Monitoring Dashboard"]
    choice = st.sidebar.selectbox("SubMenu", submenu)

    if 'df' not in st.session_state:
        st.session_state.df = None

    if choice == "Upload Data File":
        st.subheader('Upload Your Dataset')
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        delimiter = st.selectbox("Select delimiter", [";", ",", "\t", " "], index=0)
        encoding = st.selectbox("Select encoding", ["utf-8", "latin1", "iso-8859-1", "cp1252"], index=0)
        
        # Critical Parameters Input
        st.markdown("### Critical Parameters")
        date_arrete_donnees = st.date_input("Select cut-off date (DATE_ARRETE_DONNEES)", 
                                           value=datetime(2025, 4, 20),
                                           min_value=datetime(2020, 1, 1),
                                           max_value=datetime(2030, 12, 31))
        seuil_retard_critique_jours = st.number_input("Critical overdue days threshold (SEUIL_RETARD_CRITIQUE_JOURS)", 
                                                    min_value=0, 
                                                    value=0, 
                                                    step=1)
        seuil_paiement_incomplet_ratio = st.number_input("Incomplete payment ratio threshold (SEUIL_PAIEMENT_INCOMPLET_RATIO)", 
                                                        min_value=0.0, 
                                                        max_value=1.0, 
                                                        value=1.00, 
                                                        step=0.01)
        window_behavior_short = st.number_input("Short behavioral window (WINDOW_BEHAVIOR_SHORT, in number of payments)", 
                                               min_value=1, 
                                               value=3, 
                                               step=1)
        window_behavior_long = st.number_input("Long behavioral window (WINDOW_BEHAVIOR_LONG, in number of payments)", 
                                              min_value=1, 
                                              value=20, 
                                              step=1)
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding, on_bad_lines='warn')
                st.write('Reduce number of rows to increase speed.')
                frac_rows = st.slider('Select percentage of data rows', 10, 100, 100)
                nrows = int(df.shape[0] * frac_rows / 100)
                df = df.sample(n=nrows, random_state=42)
                df = preprocess_dataset(df)
                # Convert date input to pandas datetime
                date_arrete_donnees = pd.to_datetime(date_arrete_donnees)
                df = load(df, date_arrete_donnees, seuil_retard_critique_jours, seuil_paiement_incomplet_ratio,
                          window_behavior_short, window_behavior_long)
                st.session_state.df = df
                joblib.dump(df, DATAPATH / 'final_dataset.pkl')
                st.dataframe(df.head())
                st.write(f'The dataset contains {df.shape[0]} rows.')
                logging.info(f"Uploaded and saved dataset with {df.shape[0]} rows to final_dataset.pkl, "
                            f"delimiter: {delimiter}, encoding: {encoding}, "
                            f"parameters: date_arrete_donnees={date_arrete_donnees}, "
                            f"seuil_retard_critique_jours={seuil_retard_critique_jours}, "
                            f"seuil_paiement_incomplet_ratio={seuil_paiement_incomplet_ratio}, "
                            f"window_behavior_short={window_behavior_short}, "
                            f"window_behavior_long={window_behavior_long}")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.info("Try adjusting the delimiter, encoding, or critical parameters.")
                logging.error(f"Error reading CSV: {e}")
        else:
            st.info("Please upload a CSV file to proceed.")

    elif choice == "Descriptive Stats":
        st.subheader('Descriptive Statistics')
        if st.session_state.df is not None:
            st.write(st.session_state.df.describe())
            logging.info("Displayed descriptive statistics.")
        else:
            st.error("No dataset available. Please upload a CSV file in 'Upload Data File'.")
            logging.error("No dataset available for descriptive stats.")

    elif choice == "Data Types":
        st.subheader('Data Types')
        if st.session_state.df is not None:
            st.write(st.session_state.df.dtypes.astype(str))
            logging.info("Displayed data types.")
        else:
            st.error("No dataset available. Please upload a CSV file in 'Upload Data File'.")
            logging.error("No dataset available for data types.")

    elif choice == "Target Distribution":
        st.subheader('Target Distribution')
        if st.session_state.df is not None:
            target_col = st.selectbox("Select target column", st.session_state.df.columns)
            if target_col:
                df = st.session_state.df
                if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                    value_counts = df[target_col].value_counts().reset_index()
                    value_counts.columns = [target_col, 'Count']
                    fig = px.bar(value_counts, x=target_col, y='Count', title=f'Distribution of {target_col}',
                                 color=target_col, text='Count')
                    fig.update_layout(xaxis_title=target_col, yaxis_title='Count', showlegend=False)
                else:
                    fig = px.histogram(df, x=target_col, nbins=30, title=f'Distribution of {target_col}',
                                       marginal='box')
                    fig.update_layout(xaxis_title=target_col, yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)
                logging.info(f"Displayed target distribution for {target_col}.")
            else:
                st.warning("Please select a target column.")
        else:
            st.error("No dataset available. Please upload a CSV file in 'Upload Data File'.")
            logging.error("No dataset available for target distribution.")

    elif choice == "Feature Distribution":
        st.subheader('Feature Distribution')
        if st.session_state.df is not None:
            df = st.session_state.df
            ids_and_target = ['N Echeance', 'Code Client', 'N compte', 'N du Dossier', 'Code Agence', 'Code Gestionnaire', 'Impaye']
            num_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in ids_and_target]
            cat_features = list(df.select_dtypes(include=['object']).columns)
            cat_features = [col for col in cat_features if col not in ['Nom Client', 'Nom Agence']]
            
            if num_features:
                num_feature = st.selectbox("Select a numeric feature", num_features)
                fig = px.histogram(df, x=num_feature, nbins=30, title=f'Distribution of {num_feature}',
                                   marginal='box')
                fig.update_layout(xaxis_title=num_feature, yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)
                logging.info(f"Displayed numeric feature distribution for {num_feature}.")
            else:
                st.warning("No numeric features found in the dataset.")
            
            if cat_features:
                cat_feature = st.selectbox("Select a categorical feature", cat_features)
                value_counts = df[cat_feature].value_counts().reset_index()
                value_counts.columns = [cat_feature, 'Count']
                fig = px.bar(value_counts, x=cat_feature, y='Count', title=f'Distribution of {cat_feature}',
                             color=cat_feature, text='Count')
                fig.update_layout(xaxis_title=cat_feature, yaxis_title='Count', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                logging.info(f"Displayed categorical feature distribution for {cat_feature}.")
            else:
                st.warning("No categorical features found in the dataset.")
        else:
            st.error("No dataset available. Please upload a CSV file in 'Upload Data File'.")
            logging.error("No dataset available for feature distribution.")

    elif choice == "Credit Monitoring Dashboard":
        st.subheader('Credit Monitoring Dashboard')
        if st.session_state.df is not None:
            df = st.session_state.df.copy()

            status_col = st.selectbox("Select status column (e.g., Impayé, Paid, Unknown)", 
                                     ['Impaye', 'Code Client', 'N du Dossier', "Date d'Echeance", 'N Echeance', 'Region', 
                                      'Type de Credit', 'Segment', 'Secteur a Risque', 'Forme Juridique'])
            status_mapping = st.multiselect(
                "Map values to Unpaid/Paid/Unknown",
                options=df[status_col].unique(),
                default=df[status_col].unique()[:3] if len(df[status_col].unique()) >= 3 else df[status_col].unique()
            )
            if len(status_mapping) >= 1:
                status_dict = {val: 'Unpaid' if i == 1 else 'Paid' if i == 0 else 'Unknown' for i, val in enumerate(status_mapping[:3])}
                df['Status'] = df[status_col].map(status_dict).fillna('Unknown')

                st.markdown("### Key Metrics for Non-Payment Monitoring")
                col1, col2, col3, col4 = st.columns(4)
                
                unpaid_rate = (df['Status'] == 'Unpaid').mean() * 100
                with col1:
                    st.metric("Unpaid Rate", f"{unpaid_rate:.2f}%")
                
                balance_col = st.selectbox("Select balance column", ['TotalEcheance', 'capital Restant Du', 'MontantCredit', 'MvtConfEch', 'SoldeMoyEch'], key='balance_col')
                total_unpaid = df[df['Status'] == 'Unpaid'][balance_col].sum() / 1000000
                with col2:
                    st.metric("Total Unpaid Balance (en Millions de FCFA)", f"{total_unpaid:,.2f}")
                
                overdue_col = st.selectbox("Select overdue days column", df.select_dtypes(include=['int64', 'float64']).columns, key='overdue_col')
                avg_overdue = df[df['Status'] == 'Unpaid'][overdue_col].mean()
                with col3:
                    st.metric("Avg Days Overdue", f"{avg_overdue:.1f}")
                
                num_defaults = (df['Status'] == 'Unpaid').sum()
                with col4:
                    st.metric("Defaulted Loans", f"{num_defaults:,}")
                
                logging.info("Displayed key metrics for credit monitoring.")

                st.markdown("### Variable Evolution by Status")
                df = categorize_variables(df)
                num_features1 = list(df.select_dtypes(include=['object']).columns)
                
                if num_features1:
                    selected_feature = st.selectbox("Select a numerical/Categorical feature to analyze", num_features1, key='feature_evolution')
                    fig = px.histogram(df, x=selected_feature, color='Status', barmode='group',
                                       title=f'Distribution of {selected_feature} (Binned) by Status')
                    fig.update_layout(xaxis_title=f'{selected_feature} Bins', yaxis_title='Count')
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info(f"Displayed binned feature evolution for {selected_feature}.")
                else:
                    st.warning("No numerical features available for evolution analysis.")

                st.markdown("### 📊Credit Behavior Tracking")
                time_col = st.selectbox("Select time-related column (e.g., date, month)", 
                                       ['N Echeance', "Date d'Echeance", 'DateMisePlace', 'Date comptable Hist'], key='time_col')
                if time_col in df.columns:
                    fig_balance = px.line(df.groupby([time_col, 'Status'])[balance_col].sum().reset_index(),
                                          x=time_col, y=balance_col, color='Status',
                                          title=f'Total Balance Over Time by Status')
                    fig_balance.update_layout(xaxis_title=time_col, yaxis_title='Total Balance')
                    st.plotly_chart(fig_balance, use_container_width=True)
                    
                    repayment_col = st.selectbox("Select repayment amount column", ['capital Restant Du'], key='repayment_col')
                    df['Repayment_Rate'] = df[repayment_col] / df[balance_col]
                    fig_repayment = px.line(df.groupby([time_col, 'Status'])['Repayment_Rate'].mean().reset_index(),
                                            x=time_col, y='Repayment_Rate', color='Status',
                                            title=f'Average Repayment Rate Over Time by Status')
                    fig_repayment.update_layout(xaxis_title=time_col, yaxis_title='Repayment Rate')
                    st.plotly_chart(fig_repayment, use_container_width=True)
                    logging.info(f"Displayed credit behavior tracking for {time_col}.")
                else:
                    st.warning("Please select a valid time-related column.")
            else:
                st.warning("Please map at least one status value.")
        else:
            st.error("No dataset available. Please upload a CSV file in 'Upload Data File'.")
            logging.error("No dataset available for credit monitoring dashboard.")
                    