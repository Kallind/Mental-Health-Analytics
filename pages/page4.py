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
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, dbscan
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage



st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.title("Data Modelling")

col1, col2 = st.columns(2)
# def kmeans(scaled_data,df,cluster_data):
#     # elbow plot
#     inertia = []
#     for i in range(1, 11):
#         kmeans = KMeans(n_clusters=i, random_state=42)
#         kmeans.fit(scaled_data)
#         inertia.append(kmeans.inertia_)


#     # Plotting the elbow plot with plotly
#     fig = px.line(x=range(1, 11), y=inertia, title='Elbow Plot', labels={'x': 'Number of Clusters', 'y': 'Inertia'})
#     st.plotly_chart(fig)

#     kmeans = KMeans(n_clusters=4, random_state=0)
#     df['Cluster'] = kmeans.fit_predict(cluster_data)

#     df1 = df.reset_index()
#     # Displaying the clustered data interactively using Plotly Express
#     fig1 = px.scatter(df1, x='Schizophrenia', y='Depressive', color='Cluster', title='Schizophrenia vs Depressive', height=600, hover_data=['Code'])
#     st.plotly_chart(fig1)
def summarize_clustering(df, variable_of_interest,cluster_column='Cluster'):
        num_clusters = df[cluster_column].nunique()
        summary_stats = df.groupby(cluster_column).agg(['mean', 'std', 'count'])
        summary = f"The dataset has been segmented into {num_clusters} clusters. Here's a summary of the data:\n\n"
        summary += summary_stats.to_string()
        summary += f"The data is: {df[[variable_of_interest[0],variable_of_interest[1] ]]}"
        return summary
def metrics_gemini(silhouette, davies_bouldin,calinski_harabasz):
        summary = f"Silhouette Coefficient: {silhouette}" +f"Davies-Bouldin Index: {davies_bouldin}" + f"Calinski-Harabasz Index: {calinski_harabasz}"
        summary += 'given are the performance metrics of clustering data. I want inference regarding the same in genral terms'
        genai.configure(api_key= 'AIzaSyADzCNl6bP8s-G4tIyeEcrwpMPEvvL2bh0')  # Assuming API key for genai
        model = genai.GenerativeModel('gemini-pro')  # Assuming 'gemini-pro' is the Gemini model

        # Call Gemini API using genai (replace with actual call based on genai's capabilities)
        inferences = model.generate_content(summary)
        # Process and format inferences (replace with logic to handle genai response)
        insights = inferences
        insights = insights.candidates[0].content.parts[0].text
        return insights
import google.generativeai as genai
import os


def generate_insights(df, variables_of_interest, cluster_column='Cluster', gemini_api_key='your_gemini_api_key_here'):
    # Summarize clustering results (unchanged)
        summary = summarize_clustering(df,variables_of_interest, cluster_column)

        # Prepare data for Gemini API (replace with your specific logic)
        data_for_gemini = summary + '.return a presentable inferece of the clustering data.'


    # Leverage genai library (assuming it interacts with Gemini)
        try:
            genai.configure(api_key= 'AIzaSyADzCNl6bP8s-G4tIyeEcrwpMPEvvL2bh0')  # Assuming API key for genai
            model = genai.GenerativeModel('gemini-pro')  # Assuming 'gemini-pro' is the Gemini model

            # Call Gemini API using genai (replace with actual call based on genai's capabilities)
            inferences = model.generate_content(data_for_gemini)

            # Process and format inferences (replace with logic to handle genai response)
            insights = inferences
            insights = insights.candidates[0].content.parts[0].text

            return insights
        except Exception as e:
            print(f"Error using genai library: {e}")
            return "Failed to generate insights using Gemini API."
def kmeans(scaled_data,df,cluster_data,k):
    # elbow plot
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
 # Plotting the elbow plot with plotly
    fig = px.line(x=range(1, 11), y=inertia, title='Elbow Plot', labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    st.plotly_chart(fig)
 
    st.write("The elbow method in the provided plot is used to determine the optimal number of clusters (k) for k-means clustering by identifying the point at which the inertia (within-cluster sum of squares) begins to decrease more slowly. Inertia dramatically drops as the number of clusters increases and then levels off, indicating diminishing returns to adding more clusters. This point of inflection is the elbow.")
    st.write("In the provided plot, the elbow point appears to be at **k = 3**. This is where the rate of decrease in inertia shifts from being rapid to more gradual, suggesting that increasing the number of clusters beyond 3 will not result in significant improvements in the variance explained by the clusters. Therefore, **k = 3** is likely the optimal number for k-means clustering according to this elbow curve.")
 
   
    kmeans = KMeans(n_clusters=k, random_state=0)
    df['Cluster'] = kmeans.fit_predict(cluster_data)
    label = kmeans.labels_
    silhouette = metrics.silhouette_score(df, label)
    davies_bouldin = metrics.davies_bouldin_score(df, label)
    calinski_harabasz = metrics.calinski_harabasz_score(df, label)
 
    x_variable = st.selectbox('Select X Variable:', options=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'], index=0)
    y_variable = st.selectbox('Select Y Variable:', options=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'], index=1)
 
    df1 = df.reset_index()
    # Displaying the clustered data interactively using Plotly Express
    fig1 = px.scatter(df1, x=x_variable, y=y_variable, color='Cluster', title='Schizophrenia vs Depressive',color_discrete_map=px.colors.sequential.Plasma, height=600, hover_data=['Code'])
    st.plotly_chart(fig1)

    #inferencing
    st.subheader('Cluster Inferencing')

    

    st.write(generate_insights(df, (x_variable,y_variable), 'Cluster', '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'))
    perf_insight = metrics_gemini(silhouette,davies_bouldin, calinski_harabasz)
    return silhouette,davies_bouldin,calinski_harabasz, perf_insight

    # Example usage, replace 'your_dataframe' with your actual DataFrame variable
    # generate_insights(your_dataframe, ['Schizophrenia', 'Depressive'], 'Cluster', 'your_openai_api_key_here')




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
    

def hierarchical_clustering(df):
    st.title('Hierarchical Clustering and Visualization')
    
    # Perform hierarchical clustering
    linked = linkage(df, method='ward')
 
    # Perform agglomerative clustering
    num_clusters = 4
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    df['Cluster'] = agg_clustering.fit_predict(df)
    labels_dB = df['Cluster'].values
    silhouette = metrics.silhouette_score(df, labels_dB)
    davies_bouldin = metrics.davies_bouldin_score(df, labels_dB)
    calinski_harabasz = metrics.calinski_harabasz_score(df,labels_dB)
    # Plot the dendrogram
    st.subheader('Hierarchical Clustering Dendrogram')
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False, ax=ax)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    st.pyplot(fig)

    x_variable = st.selectbox('Select X Variable:', options=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'], index=0)
    y_variable = st.selectbox('Select Y Variable:', options=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'], index=1)
 
 
    # Display the clustered data interactively using Plotly Express
    st.subheader('Clustered Data Visualization')
    fig = px.scatter(df, x=x_variable, y=y_variable, color='Cluster', title=f'{x_variable} vs {y_variable}', height=600)
    st.plotly_chart(fig)
    insight = generate_insights(df, (x_variable,y_variable), 'Cluster', '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')
    st.write(insight)
    perf_insight = metrics_gemini(silhouette,davies_bouldin, calinski_harabasz)

    return silhouette,davies_bouldin,calinski_harabasz,perf_insight


##DBScan
def dbscan(df, best_eps, best_min_samples):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    df['DBCluster'] = dbscan.fit_predict(df)
    labels_dB = df['DBCluster'].values
    silhouette = metrics.silhouette_score(df, labels_dB)
    davies_bouldin = metrics.davies_bouldin_score(df, labels_dB)
    calinski_harabasz = metrics.calinski_harabasz_score(df,labels_dB)

    x_variable = st.selectbox('Select X Variable:', options=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'], index=0)
    y_variable = st.selectbox('Select Y Variable:', options=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'], index=1)
 
 
    # Display the clustered data interactively using Plotly Express
    st.subheader('Clustered Data Visualization')
    fig = px.scatter(df, x=x_variable, y=y_variable, color='DBCluster', title=f'{x_variable} vs {y_variable}', height=600)
    insight = generate_insights(df, (x_variable,y_variable), 'DBCluster', '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')
    st.plotly_chart(fig)
    st.write(insight)

    

    perf_insight = metrics_gemini(silhouette,davies_bouldin, calinski_harabasz)
    return silhouette,davies_bouldin,calinski_harabasz, perf_insight


    

    
with col1: 

    st.header("Select one from below")

    # Define options for the dropdown
    options = ["KMeans", "Hierarchical", "DBSCAN"]

    # Create the dropdown widget
    selected_option = st.selectbox("Select an option:", options)

    if selected_option == "KMeans":
        st.write(f"Selected Algorithm: {selected_option}")
        k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3, step=1)
        st.write(f"Number of Clusters (k): {k}")
        scaled_df,df,cluster_data=preprocessing()
        silhouette,davies_bouldin,calinski_harabasz, inference=kmeans(scaled_df,df,cluster_data,k)
        st.title("Performance Metrics")
        st.write(f"Silhouette Coefficient: {silhouette}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
        st.write(f"Inference\n {inference}")
        
    elif selected_option == "DBSCAN":
        eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=10, value=5, step=1)
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Epsilon (eps): {eps}")
        st.write(f"Minimum Samples: {min_samples}")
        scaled_df,df,cluster_data=preprocessing()
        silhouette,davies_bouldin,calinski_harabasz,inference= dbscan(df, eps, min_samples)
        st.title("Performance Metrics")
        st.write(f"Silhouette Coefficient: {silhouette}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
        st.write(f"Inference\n {inference}")
        # Call function to run DBSCAN algorithm with selected parameters

    elif selected_option == "Hierarchical":
        st.write(f"Selected Algorithm: {selected_option}")
        scaled_data,df,cluster_data=preprocessing()
        silhouette,davies_bouldin,calinski_harabasz, inference = hierarchical_clustering(df)
        # Display the selected option
        st.title("Performance Metrics")
        st.write(f"Silhouette Coefficient: {silhouette}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")

        st.write(f"Inference\n {inference}")

    if st.button("Testing Button"):
        switch_page("page5")

