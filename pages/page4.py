import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.title("Data Modelling")

col1, col2 = st.columns(2)
def kmeans(scaled_data,df,cluster_data):
    # elbow plot
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)


    # Plotting the elbow plot with plotly
    fig = px.line(x=range(1, 11), y=inertia, title='Elbow Plot', labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    st.plotly_chart(fig)

    kmeans = KMeans(n_clusters=4, random_state=0)
    df['Cluster'] = kmeans.fit_predict(cluster_data)

    df1 = df.reset_index()
    # Displaying the clustered data interactively using Plotly Express
    fig1 = px.scatter(df1, x='Schizophrenia', y='Depressive', color='Cluster', title='Schizophrenia vs Depressive', height=600, hover_data=['Code'])
    st.plotly_chart(fig1)

def preprocessing():
    st.title('Data Preprocessing')

    # Load data
    df1 = pd.read_csv("mental_cleaned.csv")

    df1_ent = df1.drop(["Entity", "Year"], axis=1)
    df = df1_ent.groupby('Code').mean()

    new_column_names = {
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating'
    }

    # Rename columns
    df.rename(columns=new_column_names, inplace=True)

    sc = StandardScaler()
    scaled_data = sc.fit_transform(df)

    cols = df.columns
    cluster_data = pd.DataFrame(scaled_data, columns=[cols], index=df.index)

    return scaled_data,df,cluster_data



with col1: 

    st.header("Select one from below")

    # Define options for the dropdown
    options = ["KMeans", "Clustering", "DBscan"]

    # Create the dropdown widget
    selected_option = st.selectbox("Select an option:", options)

    if selected_option == "KMeans":
        k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=3, step=1)
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Number of Clusters (k): {k}")
        scaled_df,df,cluster_data=preprocessing()
        kmeans(scaled_df,df,cluster_data)
    

    elif selected_option == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.sidebar.slider("Minimum Samples", min_value=1, max_value=10, value=5, step=1)
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Epsilon (eps): {eps}")
        st.write(f"Minimum Samples: {min_samples}")
        # Call function to run DBSCAN algorithm with selected parameters

    elif selected_option == "Hierarchical":
        linkage = st.sidebar.selectbox("Linkage Method", ["single", "complete", "average"])
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Linkage Method: {linkage}")

        # Display the selected option
        st.write("You selected:", selected_option)

        progress_bar = st.progress(0)

        # Simulate a task that takes time
    for percent_complete in range(101):
            time.sleep(0.05)  # Simulate a delay
            progress_bar.progress(percent_complete)

    with col2:
        st.write("Model Performance")
        st.write("Graphs")

    if st.button("Testing Button"):
        switch_page("page5")

