import streamlit as st
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from utils import get_categorical_feature_names_encoded, \
                  select_id_from_dataframe, \
                  get_shap_values_list, \
                  get_sorted_mean_shap_values, \
                  get_sorted_features_from_mean_shap_values    
import matplotlib.pyplot as plt
import shap


rootdir = os.getcwd()
DATAPATH = Path(rootdir) / "data"
MODELPATH = Path(rootdir) / "model"

def run_explain_app():
    # load a sample of the testing dataset
    with open(DATAPATH / 'df_test.pkl','rb') as f:
        df_test = joblib.load(f)
    # load features
    with open(MODELPATH / 'num_features.pkl','rb') as f:
        num_features = joblib.load(f)
    with open(MODELPATH / 'cat_features.pkl','rb') as f:
        cat_features = joblib.load(f)
        
    # load trained model
    try:
        with open(MODELPATH / 'pipeline.pkl','rb') as f:
            pipeline = joblib.load(f)
            # get encoded categorical feature names
            cat_features_enc = get_categorical_feature_names_encoded(pipeline, 'onehotencode', cat_features)
            # feature names after hotencoding 
            feature_names = num_features + cat_features_enc
            # get relevant feautures 
            relevant_features = pipeline['rfe'].get_feature_names_out(feature_names)
    except FileNotFoundError as e:
        st.error("""Please train the model first.""")

    st.write('Explaining the following test dataset')
    st.dataframe(df_test)

    # calc shap values
    if st.button('calculate shapley values'):
        with st.spinner('calculating shapley values...'):        

            # take random sample
            df_test = df_test.sample(250)
            # init shap explainer
            explainer = shap.explainers.Permutation(model = pipeline['rfe'].predict,
                                                    masker = pipeline['preprocessor'].transform(df_test), 
                                                    feature_names = num_features + cat_features_enc,
                                                    max_evals=1000)
            # calculate shapley values
            shap_values = explainer(pipeline['preprocessor'].transform(df_test)) 

            # save shapley values
            with open(MODELPATH / 'shap_values.pkl','wb') as f:
                joblib.dump(shap_values, f)

    # show global feature importance
    if st.button('show shap bar plot'):
        # load shapley values
        try:
            with open(MODELPATH / 'shap_values.pkl','rb') as f:
                shap_values = joblib.load(f)
        except FileNotFoundError as e:
            st.error("""Please calculate the shapley values first.""")
        
        # bar plot 
        shap.plots.bar(shap_values,
                        max_display=len(relevant_features), 
                        show=False)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title('Feature importance')
        st.pyplot(fig)


    if st.button('show shap beeswarm plot'): 
        # load shapley values
        try:
            with open(MODELPATH / 'shap_values.pkl','rb') as f:
                shap_values = joblib.load(f)
        except FileNotFoundError as e:
            st.error("""Please calculate the shapley values first.""")
        
        # bar plot 
        shap.plots.beeswarm(shap_values,
                            max_display=len(relevant_features), 
                            show=False)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title('Feature importance')
        st.pyplot(fig)



    # get relevant features from fitted pipeline  
    feat = st.selectbox("Select feature for partial dependence plot:", num_features+cat_features_enc)
    
    if st.button('show partial dependence plot'):
        # load shapley values
        try:
            with open(MODELPATH / 'shap_values.pkl','rb') as f:
                shap_values = joblib.load(f)
        except FileNotFoundError as e:
            st.error("""Please calculate the shapley values first.""")
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.gca()
        # take random sample
        df_test = df_test.sample(250)
        shap.dependence_plot(feat, 
                            shap_values = shap_values.values, 
                            features = pd.DataFrame(data = pipeline['preprocessor'].transform(df_test), 
                                                    columns = num_features + cat_features_enc),
                            x_jitter = 0.2,
                            xmin="percentile(2.5)",
                            xmax="percentile(97.5)", 
                            interaction_index=None,
                            title = 'SHAP Dependence Plot: SHAP Value vs {}'.format(feat),
                            ax=ax,
                            show=False)
        ax.grid('on')
        st.pyplot(fig)

    

    # show local explanbility i.e. explain single row  
    ID = st.selectbox('Select ID to be explained:', df_test['Nom Client'].values) #N du Dossier
    st.write('explaining ID (N_ du Dossier): {}'.format(ID)) 
    row_selected =  select_id_from_dataframe(df_test, ID)
    st.write('Predicted default probability for selection: {:.2f}'.format(pipeline.predict_proba(row_selected)[0][1]))
    st.dataframe(row_selected)
    if st.button('explain single decision:'):
        # generate list of shap values using random data samples
        shap_values_list = get_shap_values_list(pipeline, feature_names, df_test, row_selected, 50)
        # sort mean and sd and feature names
        shap_values_mean_sorted, shap_values_sd_sorted = get_sorted_mean_shap_values(shap_values_list)
        feature_names_sorted = get_sorted_features_from_mean_shap_values(shap_values_list)

        fig, ax = plt.subplots(1,1,figsize=(8,6))
        y = feature_names_sorted[0:len(relevant_features)][::-1]
        x = shap_values_mean_sorted[0:len(relevant_features)][::-1]
        xerr=shap_values_sd_sorted[0:len(relevant_features)][::-1]
        colors = ['b' if e >= 0 else 'r' for e in shap_values_mean_sorted[0:len(relevant_features)][::-1]]
        ax.barh(y=y, 
                width=x,
                xerr=xerr,
                color=colors)
        ax.grid()
        st.pyplot(fig)

