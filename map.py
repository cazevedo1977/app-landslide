import streamlit as st
import pandas as pd
import numpy as np
import os

import Rede_Neural as rna
from Rede_Neural import NeuralNetwork
from Rede_Neural import Layer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif



import folium
from folium import plugins
from streamlit_folium import st_folium
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import utm



APP_TITLE = 'Machine Learning for Landslide Susceptibility'
APP_SUB_TITLE = 'From Theory to Practice: Machine Learning for Landslide Susceptibility Assessment in the Brazilian Atlantic Forest'


#Get data with UTM coordinates and convert them to lat/long
@st.cache_data
def getdata(fileName=None):
    landslide_df = pd.read_csv(fileName)
    zone_number = 23   #São Paulo coast UTM zone number
    zone_letter = 'L'  #São Paulo coastUTM zone letter
    landslide_df['lat'], landslide_df['lon'] = utm.to_latlon(easting=landslide_df['X'].values,northing= landslide_df['Y'].values,zone_number=zone_number,zone_letter=zone_letter)
    
    print("getdata " + fileName)
    return landslide_df


# Prepare and format data to perform prediction based on user selection
def dataPreparation(dataSet):
    
    dataSet = dataSet[['slope','aspect','elevation','land_use','lithology','twi','curvature','class','lat','lon']]
    y = dataSet['class']
    
    # split data into training, validation and testing sets
    test_size = 0.30
    train_ds, test_ds, train_dsy, test_dsy = train_test_split(dataSet, y, test_size=test_size, shuffle=False)
    train_ds, val_ds, train_dsy, val_dsy   = train_test_split(train_ds,train_dsy, test_size=test_size, shuffle=False)

    
    train_x  = train_ds
    train_x  = train_x.drop(['class','lat','lon'],axis=1)
    
    test_x  = test_ds
    test_x  = test_x.drop(['class','lat','lon'],axis=1)

    
    val_x  = val_ds
    val_x  = val_x.drop(['class','lat','lon'],axis=1)

    train_y = train_dsy.to_numpy().reshape(-1,1)
    test_y = test_dsy.to_numpy().reshape(-1,1)
    val_y = val_dsy.to_numpy().reshape(-1,1)

    
    # define the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    #scaler = MinMaxScaler() 
    # fit on the training dataset
    scaler.fit(train_x)
    # scale the training dataset
    train_x = scaler.transform(train_x)
    # scale the test dataset
    test_x = scaler.transform(test_x)
    # scale the test dataset
    val_x = scaler.transform(val_x)
    
    if st.session_state.sample == "train":
        dataSet = train_ds
        X = train_x
        y = train_y
    elif st.session_state.sample == "test":
        dataSet = test_ds
        X = test_x
        y = test_y
    elif st.session_state.sample == "validation":
        dataSet = val_ds
        X = val_x
        y = val_y
    else:
        X = np.concatenate((train_x, val_x))
        X = np.concatenate((X, test_x))
        y = np.concatenate((train_y, val_y))
        y = np.concatenate((y, test_y))

    y = y.ravel() #convert that array shape to (n, ) (i.e. flatten it)    

    #print("data prep " + st.session_state.sample)
    return X, y, dataSet

# Prepare and format data to perform prediction based on user selection
def dataPreparation2(dataSet):
    
    dataSet = dataSet[['slope','aspect','elevation','land_use','lithology','twi','curvature','class','lat','lon']]
    X = dataSet.drop('class',axis=1)
    X = X.drop('lat',axis=1)
    X = X.drop('lon',axis=1)
    y= dataSet['class']
    
    X = X.to_numpy()    #converts dataframe into array to be used by the  NN
    y = y.to_numpy()    #converts dataframe into array to be used by the  NN
    y = y.reshape(-1,1) #reorganiza o array em um array 1 x 1
    
    # split data into training, validation and testing sets
    test_size = 0.30
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=False)
    train_x, val_x, train_y, val_y   = train_test_split(train_x,train_y, test_size=test_size, shuffle=False)

    train_ds, test_ds, train_dsy, test_dsy = train_test_split(dataSet, y, test_size=test_size, shuffle=False)
    train_ds, val_ds, train_dsy, val_dsy   = train_test_split(train_ds,train_dsy, test_size=test_size, shuffle=False)

    # define the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    #scaler = MinMaxScaler() 
    # fit on the training dataset
    scaler.fit(train_x)
    # scale the training dataset
    train_x = scaler.transform(train_x)
    # scale the test dataset
    test_x = scaler.transform(test_x)
    # scale the test dataset
    val_x = scaler.transform(val_x)
    
    if st.session_state.sample == "train":
        dataSet = train_ds
        X = train_x
        y = train_y
    elif st.session_state.sample == "test":
        dataSet = test_ds
        X = test_x
        y = test_y
    elif st.session_state.sample == "validation":
        dataSet = val_ds
        X = val_x
        y = val_y
    else:
        X = np.concatenate((train_x, val_x))
        X = np.concatenate((X, test_x))
        y = np.concatenate((train_y, val_y))
        y = np.concatenate((y, test_y))
    
    y = y.ravel() #convert that array shape to (n, ) (i.e. flatten it)    

    #print("data prep")
    return X, y, dataSet

##### predictions functions ##################
#Load trained ANN and predict landslide susceptibility 
def run_prediction(dS,features):
    bestNN = None 
    bestNN = NeuralNetwork.load('customNN_guaruja_random.pkl')
    y_pred = bestNN.predict(features)
    dS['score'] = y_pred
    dS['prediction'] = dS['score'].apply(lambda x: 1 if x > 0.5 else 0)
    return dS, y_pred


#Compute feature importance based on ANOVA/F-test score.
def features_importance(X, y):
    # Setup KBestFeatures class. k=5 most important features
    fs = SelectKBest(score_func=f_classif, k=5)
    # Apply feature selection
    fs.fit_transform(X, y)
    predictors = X.columns
    #fs.pvalues_  | fs.scores_
    scores = -np.log10(fs.pvalues_) #or scores /= scores.max()
    scores = np.round(scores, 2)
    scores = (scores/scores.max())*10 #podemos normalizar só para manter a mesma escala.

    return predictors, scores 

# Compute and display performance metrics 
def NetworkPerformance(y_real, y_prob):
    y_predict = (y_prob > 0.5)
    accuracy = accuracy_score(y_real, y_predict) # accuracy: (tp + tn) / (p + n)
    #print('Accuracy: %f' % accuracy)
    precision = precision_score(y_real, y_predict)     # precision tp / (tp + fp)
    #print('Precision: %f' % precision)
    recall = recall_score(y_real, y_predict) # recall: tp / (tp + fn)
    #print('Recall: %f' % recall)
    f1 = f1_score(y_real, y_predict) # f1: 2 tp / (2 tp + fp + fn)
    #print('F1 score: %f' % f1)
    # ROC AUC
    auc = roc_auc_score(y_real, y_prob)
    #print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_real, y_predict)
    return accuracy, recall, auc 

##### end predictions functions ##################

##### sidebar user interface functions ##################

def file_selector(folder_path=os.path.abspath(os.getcwd())):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox(':open_file_folder: Select a file: ', filenames)
    return os.path.join(folder_path, selected_filename)

def display_user_interaction():
    path = os.path.join(os.path.abspath(os.getcwd()), 'data')
    filename = file_selector(folder_path=path)
    
    st.sidebar.selectbox(
        'Map Style :world_map:',
        ('OpenStreetMap', 'Cartodb dark_matter', 'CartoDB positron', 'OpenTopoMap'),
        key='map_style',
        index = 0
        )

    #st.sidebar.checkbox("Disable selectbox widget", key="disabled")
    st.sidebar.radio(
        "Set dataset sample:",
        key="sample",
        options=["train", "test", "validation","full"],
        index=0
    )

    return filename
##### end sidebar user interface functions ##################

##### metrics, charts and maps functions ##################
def plot_map(df):
    
    map = folium.Map(location=[df['lat'].values.mean(), df['lon'].values.mean()], zoom_start=11.5, width='100%',height="100", scrollWheelZoom=True, tiles=st.session_state.map_style)
    minimap = plugins.MiniMap()
    map.add_child(minimap)
    for i in range(0,len(df)):

        
        if 0 <= df.iloc[i]['score'] < 0.5:
            color = 'green'
            classification = 'low'
        elif 0.5 <= df.iloc[i]['score'] < 0.9:
            color = 'yellow'
            classification = 'moderate'
        else:
            color = 'red'
            classification = 'high'

        prediction_accuracy = "false"
        
        right = f"""<rect x="7", y="7" width="5" height="5", fill={color}, opacity=".3" /> """
        if df.iloc[i]['class'] == df.iloc[i]['prediction']:
            prediction_accuracy = "true"
            right = f""""""


        html=f"""
            <div style="width: 165px; height=210px; z-index: 2; border-radius: 10px; border-radius: 4px; color: white; font-size: 12px; background-color: #192733; font-family: courier new">
            <h5> ANN classification: {classification}</h5>
            <p style="font-size:10px">Additional information:</p>
            <ul>
                <li style="font-size:10px">actual: {df.iloc[i]['class']}</li>
                <li style="font-size:10px">prediction: {df.iloc[i]['prediction']}</li>
                <li style="font-size:10px">rightnesses: {prediction_accuracy}</li>
                <li style="font-size:10px">probability: {'{:.2f}%'.format(df.iloc[i]['score']*100)}</li>
            </ul>
            <ul>
                <li style="font-size:10px">slope: {'{:.3f}'.format(df.iloc[i]['slope'])}</li>
                <li style="font-size:10px">twi: {'{:.3f}'.format(df.iloc[i]['twi'])}</li>
                <li style="font-size:10px">aspect: {'{:.3f}'.format(df.iloc[i]['aspect'])}</li>
                <li style="font-size:10px">elevation: {'{:.3f}'.format(df.iloc[i]['elevation'])}</li>
                <li style="font-size:10px">curvature: {'{:.3f}'.format(df.iloc[i]['curvature'])}</li>
                
            </ul>
            </p>
            </div>
            """
        
        iframe = folium.IFrame(html=html, width=175, height=210)
        popup = folium.Popup(iframe, max_width=2650)
        folium.Marker(
            location=[df.iloc[i]['lat'], df.iloc[i]['lon']],
            #tooltip=folium.Tooltip(iframe.render()),
            popup=popup,
            icon=folium.DivIcon(html=f"""
                <div><svg>
                    <circle cx="10" cy="10" r="8" fill={color} opacity=".4"/>
                    {right}
                </svg></div>""")
        ).add_to(map)

    return map


def plot_classification_pie(source):

    base = alt.Chart(source).encode(
        theta=alt.Theta("total:Q", stack=True), 
        color=alt.Color("classification:N", legend=None)
    ).properties(
        title=f"Prediction classification over {(source['total'].sum())} pinpoints"
    )
    pie = base.mark_arc(outerRadius=100)
    text = base.mark_text(radius=140, size=18).encode(text="classification:N")

    chart = pie + text
    return chart


def plot_bar_ann_assessment(source):

    chart = alt.Chart(source).mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4
    ).encode(
        x='total:Q',
        y='classification:O'
        
    ).properties(title="ANN assessment")
    return chart
     

def plot_horizontal_bar(source):

    chart = alt.Chart(source).mark_bar().encode(
        x='scores:Q',
        y="predictors:O"
    ).properties(title="Feature importance")

    return chart 
    
##### end metrics, charts and maps functions ##################



def main():
    st.theme = "dark"
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)
    st.markdown("""
            This app performs landslide susceptibility prediction based on data from Guarujá - Brazil!
        * **Python libraries:** in-house neural network, pandas, streamlit
        * **Code and data sources:** [Caio Azevedo Github](https://github.com/cazevedo1977/academico/tree/main/doutorado/tese/paper_susceptibility_map) & **Theoretical Foundation** [Masters Dissertation](https://www.teses.usp.br/teses/disponiveis/3/3136/tde-11052022-103227/publico/CaiodaSivaAzevedoCorr22.pdf).
        """)
    
    #filename =  display_user_interaction()
    
    #Load and Prepare Data depending on user selected data file and sample
    df = getdata(fileName = display_user_interaction())
    X, y, df = dataPreparation(df)
    
    #Predict landslide based on prepared data and return dataset with computed columns
    df, y_pred  = run_prediction(df,X)
    acc, rcl, auc = NetworkPerformance(y, (y_pred))

    #prepare a dataframe to compute features importance
    df_importance = df[['slope','aspect','elevation','land_use','lithology','twi','curvature']]
    predictors, scores = features_importance(X=df_importance, y=y)
    
    #Display Metrics
    st.subheader(f'Predictions Performance')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accurancy", value='{:.2f}%'.format(acc*100), help="It is the fraction of predictions our model got right, it is computed by: (tp + tn) / (tp + tn + fp + fn)" )
    with col2:
        st.metric("Sensitivity", value='{:.2f}%'.format(rcl*100), help="Also known as Recall or True positive rate it is computed by: tp / (tp + fn)")
    with col3:
        st.metric("ROC-AUC", value='{:.2f}%'.format(auc*100), help="It correlates the true positive rate (TPR) against the false positive rate (FPR)")

     ## Create a map with a specified zoom level
    col4, col5, col6 = st.columns(3)
    
    ann_prediction_data = pd.DataFrame({'classification': ['right', 'wrong'], 'total': [len(df[df['class'] == df['prediction']]),len(df[df['class'] != df['prediction']])]})

    pie_data = pd.DataFrame({'classification': ['high', 'moderate', 'low'], 
                             'total': [len(df[df['score']>=0.9]),len(df[(df["score"]>= 0.5) & (df["score"]<=0.9)]),len(df[df['score']<0.5])]})
    
    chart_data = pd.DataFrame({'predictors':predictors, 'scores':scores})

    chart_classification = plot_classification_pie(source=pie_data)
    chart_features_importance = plot_horizontal_bar(source=chart_data)
    chart_prediction_assessment = plot_bar_ann_assessment(source=ann_prediction_data)

    
    col4.altair_chart(chart_classification, theme="streamlit", use_container_width=True)
    col5.altair_chart(chart_features_importance, theme="streamlit", use_container_width=True)
    col6.altair_chart(chart_prediction_assessment, theme="streamlit", use_container_width=True)

    map = plot_map(df=df)    
    st_folium(fig=map, width=1200,height=500)

    st.write(df,unsafe_allow_html=True)

if __name__ == "__main__":
    main()

#references:
# https://altair.streamlit.app/ (biblioteca excelente de gráficos)
# https://docs.kanaries.net/topics/Streamlit/streamlit-map
# https://www.countrycoordinate.com/city-guaruja-brazil/
# https://github.com/opengeos/streamlit-geospatial/tree/master
# https://github.com/gee-community/geemap/issues/713
# https://github.com/opengeos/streamlit-geospatial/tree/master?tab=readme-ov-file
# https://stackoverflow.com/questions/71130194/switching-off-all-folium-tiles     
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/    
# https://landslides.streamlit.app/ (url da aplicação)
# https://github.com/randyzwitch/streamlit-folium/issues/7 (sobre mapa com tamanho total da tela)    
# https://fabiobenevides.medium.com/planejando-e-visualizando-informa%C3%A7%C3%B5es-de-sua-viagem-utilizando-python-folium-e-google-912bce9e109c (sobre tooltip do mapa)
# https://stackoverflow.com/questions/62789558/is-it-possible-to-change-the-popup-background-colour-in-folium (sobre tooltip do mapa)
# https://python-graph-gallery.com/312-add-markers-on-folium-map/
# https://discuss.streamlit.io/t/modulenotfounderror-no-module-named-sklearn/48314/5 (sobre a instalação das libs no server streamlit)    
# https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies (sobre a instalação das libs no server streamlit)    
# https://www.askpython.com/python-modules/streamlit-theming (Theme)