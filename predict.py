# import modules
import os
from pathlib import Path
import joblib
import pandas as pd

# set paths
rootdir = os.getcwd()
DATAPROCESSEDPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'


# load training dataset
df = pd.read_csv(DATAPROCESSEDPATH / 'data_preprocessed.csv')

# load pipeline
with open(MODELPATH / 'pipeline.pkl','rb') as f:
    pipe_loaded = joblib.load(f)

print(pipe_loaded)

## select first row
#df = df[0:1]

# seperate into numerical and categorical features
num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
cat_features = list(df.dtypes[df.dtypes == 'object'].index)
label = ['Impaye']
print('-----------------------------------')
# Définir les colonnes numériques et catégoriques
num_features = ['TotalEcheance', 'capital Restant Du', 'NbreEch', 'TauxInteret', 'MontantCredit', 'AgeRelationBqe', 
                'AncienneteActivite', 'MvtConfEch', 'nb_incidents_paiement_pass_short_win', 
                'nb_incidents_paiement_pass_long_win', 'retard_moyen_pass_short_win', 
                'retard_moyen_pass_long_win', 'ratio_paiement_moyen_pass_short_win', 
                'jours_depuis_dernier_incident', 'nb_impayes_cumul']
cat_features = ['Type de Credit_G', 'Segment_G', 'Secteur a Risque_G', 'FormeJuridique_G', 'Region']

print('numeric features: \n', num_features)
print('-----------------------------------')
print('categorical features: \n', cat_features)

# get feature array
X = df[num_features + cat_features]

# make prediction
y_scores = pipe_loaded.predict_proba(X)
print('predicted credit default scores: ', y_scores[:,1])

# append predictions to df
df['predicted credit default scores'] = y_scores[:,1]
print(df.head())