import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px


import Rede_Neural as rna
from Rede_Neural import NeuralNetwork
from Rede_Neural import Layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

def features_importance(X, y):
    # Setup KBestFeatures class. k=5 most important features
    fs = SelectKBest(score_func=f_classif, k=5)
    # Apply feature selection
    fs.fit_transform(X, y)
    predictors = X.columns
    #fs.pvalues_  | fs.scores_
    scores = -np.log10(fs.pvalues_) #or scores /= scores.max()
    scores = np.round(scores, 2)
    return predictors, scores 

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

     
@st.cache_data
def plot_horizontal_bar(source):

    chart = alt.Chart(source).mark_bar().encode(
        x=alt.X(field="predictors", title=None),
        y=alt.Y(
            "scores:N",
            sort=alt.Sort(field="scores", order="descending"),
            title=None,
        ),
    ).transform_window(
        rank='row_number()',
        sort=[alt.SortField("scores", order="descending")],
    ).properties(
        title="Feature importance",
    )

    return chart 
    



def file_selector(folder_path='.\data'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox(':open_file_folder: Select a file: ', filenames)
    return os.path.join(folder_path, selected_filename)

def getdata(fileName=None):
    #Get data with UTM coordinates and convert them to lat/long
    landslide_df = pd.read_csv(fileName)
    zone_number = 23   #São Paulo coast UTM zone number
    zone_letter = 'L'  #São Paulo coastUTM zone letter
    landslide_df['lat'], landslide_df['lon'] = utm.to_latlon(easting=landslide_df['X'].values,northing= landslide_df['Y'].values,zone_number=zone_number,zone_letter=zone_letter)
    
    return landslide_df

def run_prediction(dS,features):
    #load trained ANN
    bestNN = None 
    bestNN = NeuralNetwork.load('customNN_guaruja_random.pkl')
    y_pred = bestNN.predict(features)
    dS['score'] = y_pred
    dS['prediction'] = [int(b) for b in (y_pred > 0.5)]
    return dS, y_pred

# split data into training, validation and testing sets
def dataPreparation(dataSet, sample):
    
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
    
    if "sample" not in st.session_state:
        st.session_state.sample = "train"
        
    match st.session_state.sample:
        case "train":
            dataSet = train_ds
        case "test":
            dataSet = test_ds
        case "validation":
            dataSet = val_ds
        case _:
            dataSet
    
    return train_x, test_x, val_x, train_y, test_y, val_y, dataSet

# Compute and display performance metrics 
def NetworkPerformance(y_real, y_prob):
    y_predict = (y_prob > 0.5)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_real, y_predict)
    #print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_real, y_predict)
    #print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_real, y_predict)
    #print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_real, y_predict)
    #print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_real, y_predict)
    #print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_real, y_prob)
    #print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_real, y_predict)
    
    return accuracy, recall, auc 


def display_user_interaction():
    filename = file_selector()
    #st.write('You selected `%s`' % filename)
    if "sample" not in st.session_state:
        st.session_state.sample = "train"
        #st.session_state.disabled = False

    option = st.sidebar.selectbox(
        'Map Style :world_map:',
        ('OpenStreetMap', 'Cartodb dark_matter', 'CartoDB positron', 'OpenTopoMap'),
        key='map_style'
        )


    #st.sidebar.checkbox("Disable selectbox widget", key="disabled")
    sample = st.sidebar.radio(
        "Set dataset sample :eyeglasses:",
        key="sample",
        options=["train", "test", "validation","full"],
        index=0
    )

    return filename, sample



def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)
    st.markdown("""
            This app performs landslide susceptibility prediction based on data from Guarujá - Brazil!
        * **Python libraries:** in-house neural network, pandas, streamlit
        * **Data source:** [Caio Azevedo Github](https://github.com/cazevedo1977/academico/tree/main/doutorado/tese/paper_susceptibility_map).
        """)
    
    filename, sample = display_user_interaction()
    
    #Load and Prepare Data
    df = getdata(fileName = filename)
    X_train, X_test, X_val, y_train, y_test, y_val, df = dataPreparation(df,sample)
    X = np.concatenate((X_train, X_val))
    X = np.concatenate((X, X_test))
    y = np.concatenate((y_train, y_val))
    y = np.concatenate((y, y_test))

    match st.session_state.sample:
        case "train":
            X = X_train
            y = y_train
        case "test":
            X = X_test
            y = y_test
        case "validation":
            X = X_val
            y = y_val
        
    y = y.ravel() #convert that array shape to (n, ) (i.e. flatten it)
    
    #Predict landslide based on prepared data and return dataset with computed columns
    df, y_pred  = run_prediction(df,X)
    acc, rcl, auc = NetworkPerformance(y, (y_pred))

    df_importance = df[['slope','aspect','elevation','land_use','lithology','twi','curvature']]
    predictors, scores = features_importance(X=df_importance,y=y)
    
    #Display Metrics
    st.subheader(f'Predictions Performance')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accurancy", '{:.2f}%'.format(acc*100))
    with col2:
        st.metric("Sensibility", '{:.2f}%'.format(rcl*100))
    with col3:
        st.metric("ROC-AUC", '{:.2f}%'.format(auc*100))

    
    map = folium.Map(location=[df['lat'].values.mean(), df['lon'].values.mean()], zoom_start=11.5, width='700',height="100", scrollWheelZoom=True, tiles=st.session_state.map_style)
    minimap = plugins.MiniMap()
    map.add_child(minimap)
    for i in range(0,len(df)):

        prediction_accuracy = "false"
        if df.iloc[i]['class'] == df.iloc[i]['prediction']:
            prediction_accuracy = "true"

        match df.iloc[i]['score']:
            case num if 0 <= num <  0.5: 
                color = 'green'
                classification = 'low'
            case num if 0.5 <= num <  0.9: 
                color = 'yellow'
                classification = 'moderate'
            case _:
                color = 'red'
                classification = 'high'


        html=f"""
            <h5> ANN classification: {classification}</h5>
            <p style="font-size:13px">additional information:</p>
            <ul>
                <li style="font-size:12px">actual: {df.iloc[i]['class']}</li>
                <li style="font-size:12px">prediction: {df.iloc[i]['prediction']}</li>
                <li style="font-size:12px">accuracy: {prediction_accuracy}</li>
                <li style="font-size:12px">probability: {'{:.2f}%'.format(df.iloc[i]['score']*100)}</li>
            </ul>
            </p>
            """
        iframe = folium.IFrame(html=html, width=175, height=168)
        popup = folium.Popup(iframe, max_width=2650)
        folium.Marker(
            location=[df.iloc[i]['lat'], df.iloc[i]['lon']],
            popup=popup,
            icon=folium.DivIcon(html=f"""
                <div><svg>
                    <circle cx="10" cy="10" r="8" fill={color} opacity=".4"/>
                </svg></div>""")
        ).add_to(map)

    st_folium(fig=map, width=1200,height=500)


     ## Create a map with a specified zoom level
    col4, col5, col6 = st.columns(3)
    
    ann_prediction_data = pd.DataFrame({'classification': ['right', 'wrong'], 'total': [len(df[df['class'] == df['prediction']]),4]})

    pie_data = pd.DataFrame({'classification': ['high', 'moderate', 'low'], 
                             'total': [len(df[df['score']>=0.9]),len(df[(df["score"]>= 0.5) & (df["score"]<=0.9)]),len(df[df['score']<0.5])]})
    
    chart_data = pd.DataFrame({'predictors':predictors, 'scores':scores})

    chart_classification = plot_classification_pie(source=pie_data)
    chart_features_importance = plot_horizontal_bar(source=chart_data)
    chart_prediction_assessment = plot_classification_pie(source=pie_data)

    
    col4.altair_chart(chart_classification, theme="streamlit", use_container_width=True)
    col5.altair_chart(chart_features_importance, theme="streamlit", use_container_width=True)
    col6.altair_chart(chart_prediction_assessment, theme="streamlit", use_container_width=True)
    
    st.write(df)

if __name__ == "__main__":
    main()

#references:
# https://docs.kanaries.net/topics/Streamlit/streamlit-map
# https://www.countrycoordinate.com/city-guaruja-brazil/
# https://github.com/opengeos/streamlit-geospatial/tree/master
# https://github.com/gee-community/geemap/issues/713
# https://github.com/opengeos/streamlit-geospatial/tree/master?tab=readme-ov-file
# https://stackoverflow.com/questions/71130194/switching-off-all-folium-tiles     
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/    
# https://landslides.streamlit.app/