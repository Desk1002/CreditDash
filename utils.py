import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
import shap

def plot_num_feature_distribution(df:pd.DataFrame, num_feature:str):
            """show distribution and boxplots for a given numeric feature."""
            fig, ax = plt.subplots(1,2, figsize=(16,6))    
            sns.histplot(data=df, 
                        x=num_feature, 
                        hue='Impaye', 
                        stat='percent', 
                        kde=True,
                        element='step',
                        ax=ax[0])
            ax[0].set_title(num_feature)
            ax[0].legend(['credit default', 'no credit default'])

            sns.boxplot(data=df, 
                        y=num_feature, 
                        x='Impaye',
                        ax=ax[1])
            ax[1].set_title(num_feature)
            ax[1].set_xticklabels(['credit default', 'no credit default'])
            return fig

def plot_cat_feature_distribution(df, cat_feature):
    """plot histogram for a given categorical feauture stratified by the target columns e.g. here Status"""
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.countplot(x=cat_feature, hue="Impaye", data=df, ax=ax)
    ax.set_title('Barplot Stratified by ' + cat_feature, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend(['not credit default', 'credit default'], fontsize=12)
    return fig

def plot_target_distribution(df, target_label:str='Impaye'):
    """plot histogram for the target variable e.g. here Status"""
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.countplot(x=target_label, data=df, ax=ax)
    ax.set_title('credit default distribution', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    return fig

def convert_df(df):
    """Convert dataframe to csv"""
    return df.to_csv(index=False).encode('utf-8')

def get_categorical_feature_names_encoded(pipeline:Pipeline, hotencoding_label:str, categorical_feature_names:list):
    """Get the names of categorical features after being hotencoded in a sklearn pipeline. 
        Pass in the pipeline object, the label of the one-hotencoding step in the pipeline. 
        Also pass in the names of the categorical feautures before the one-hot-encoding.
        Watchout: the code is little instable since the hard-coded transformers list: transformers_[1][1]. U might need to adapt this!"""
    return list(pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps[hotencoding_label].get_feature_names_out(categorical_feature_names))    

def select_row_from_dataframe(dataframe:pd.DataFrame, row:int)->pd.DataFrame:
    """Select single row from dataframe for model explainability given a row number."""
    return dataframe.iloc[np.arange(row,row+1),:]

def select_id_from_dataframe(dataframe:pd.DataFrame, id:int)->pd.DataFrame:
    """Select single row from dataframe for model explainability given a particular ID"""
    return dataframe.loc[dataframe['Nom Client']==id,:]  #N du Dossier 'Nom Client'

def get_sorted_shap_values(shap_values, order='descending')->np.array:
    """get sorted shap_values from shap values input. Sorted mean in descending order using the absolute value of shap values."""
    shap_values_sorted = [shap_values[0].values[ind] for ind in np.argsort(np.abs(shap_values[0].values), kind='stable')]
    if order == 'ascending':
        return shap_values_sorted
    elif order == 'descending':
        return shap_values_sorted[::-1]
    else:
        print('check order specification')

def get_sorted_features_from_shap_values(shap_values, order='descending')->list:
    """get sorted feature names from shap values input. Sorted mean in descending order using the absolute shap values."""
    feature_names_sorted = [shap_values.feature_names[feature_ind] for feature_ind in np.argsort(np.abs(shap_values[0].values), kind='stable')]
    if order == 'ascending':
        return feature_names_sorted
    elif order == 'descending':
        return feature_names_sorted[::-1]
    else:
        print('check order specification')

def get_shap_values_list(pipeline, feature_names, dataframe:pd.DataFrame, row_selected:pd.DataFrame, number_random_samples:int=20)->list:
    """Get a list of shap values by randomly drawing samples from the input dataframe and calculating shap values for a single data row.
        It is assumed that the trained pipeline is inputted with the model being accessable using pipeline['rfe] and the data transformation
        being accessable using pipeline['preprocessor'].transform(data). Also provide a list of the feature names outputted by the pipeline."""
    shap_values_list = [] # collect n shap values for calculating the mean and standard deviation
    for n in range(number_random_samples):
        # get data sample
        data = dataframe.sample(100, replace=True) 
        # init shap explainer
        explainer = shap.explainers.Permutation(model=pipeline['rfe'].predict,
                                                masker=pipeline['preprocessor'].transform(data), 
                                                feature_names=feature_names,
                                                max_evals=1000)
        # calculate shapley values
        shap_values = explainer(pipeline['preprocessor'].transform(row_selected))
        shap_values_list.append(shap_values)
    return shap_values_list        

def get_mean_from_shap_value_list(shap_values_list):
    return np.array([shap_values_list[n][0].values for n in range(len(shap_values_list))]).mean(axis=0)

def get_sd_from_shap_value_list(shap_values_list):
    return np.array([shap_values_list[n][0].values for n in range(len(shap_values_list))]).std(axis=0)        

def get_sorted_mean_shap_values(shap_values_list:list, order='descending')->np.array:
    """Get sorted shap_values inputing the shap_value_list from the method "get_shap_values_list". 
        Sorted mean in descending order using the absolute value of shap values.
        The sorted mean and the sorted standard deviation are returned """
    # calculate mean values and standard deviation
    shap_values_mean = get_mean_from_shap_value_list(shap_values_list)
    shap_values_sd = get_sd_from_shap_value_list(shap_values_list)
    # sorte shap values 
    shap_values_mean_sorted = [shap_values_mean[ind] for ind in np.argsort(np.abs(shap_values_mean), kind='stable')]
    shap_values_sd_sorted = [shap_values_sd[ind] for ind in np.argsort(np.abs(shap_values_mean), kind='stable')]
    if order == 'ascending':
        return shap_values_mean_sorted, shap_values_sd_sorted
    elif order == 'descending':
        return shap_values_mean_sorted[::-1], shap_values_sd_sorted[::-1]
    else:
        print('check order specification')

def get_sorted_features_from_mean_shap_values(shap_values_list:list, order='descending')->np.array:
    """get sorted feature names inputing the shap_value_list from the method "get_shap_values_list"."""
    # calculate mean values and standard deviation
    shap_values_mean = get_mean_from_shap_value_list(shap_values_list)
    # sort the feature names
    feature_names_sorted = [shap_values_list[0].feature_names[feature_ind] for feature_ind in np.argsort(np.abs(shap_values_mean), kind='stable')]
    if order == 'ascending':
        return feature_names_sorted
    elif order == 'descending':
        return feature_names_sorted[::-1]
    else:
        print('check order specification')
        


###############################################################################
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

#################################################################################################
