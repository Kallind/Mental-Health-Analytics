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
    return silhouette,davies_bouldin,calinski_harabasz

    #inferencing
    st.subheader('Cluster Inferencing')
    import pandas as pd

    def summarize_clustering(df, cluster_column='Cluster'):
        num_clusters = df[cluster_column].nunique()
        summary_stats = df.groupby(cluster_column).agg(['mean', 'std', 'count'])
        summary = f"The dataset has been segmented into {num_clusters} clusters. Here's a summary of the data:\n\n"
        summary += summary_stats.to_string()
        summary += f"The data is: {df}"
        return summary

    # def create_chatgpt_prompt(data_summary, variables_of_interest):
    #     prompt = f"I have performed a clustering analysis focusing on variables such as {', '.join(variables_of_interest)}. {data_summary} Based on this clustering, what insights or patterns can we deduce? What further analysis would you recommend?"
    #     return prompt
    
    # from openai import OpenAI

    # def call_chatgpt_api(prompt, api_key):
    #     client = OpenAI(api_key=api_key)

    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "system", "content": "You are an intelligent data analyst."},
    #             {"role": "user", "content": prompt}
    #         ]
    #     )
    #     return response.choices[0].message.content

    
    # def generate_insights(df, variables_of_interest, cluster_column='Cluster', api_key='your_openai_api_key_here'):
    # # Summarize clustering results
    #     summary = summarize_clustering(df, cluster_column)
        
    #     # Create a prompt for ChatGPT
    #     prompt = create_chatgpt_prompt(summary, variables_of_interest)
        
    #     # Make an API call to ChatGPT
    #     insights = call_chatgpt_api(prompt, api_key)
        
    #     return f"Generated Insights:\n{insights}"

    import google.generativeai as genai
    import os


    def generate_insights(df, variables_of_interest, cluster_column='Cluster', gemini_api_key='your_gemini_api_key_here'):
    # Summarize clustering results (unchanged)
        summary = summarize_clustering(df, cluster_column)

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

            return f"Generated Insights based on Gemini Analysis:\n{insights}"
        except Exception as e:
            print(f"Error using genai library: {e}")
            return "Failed to generate insights using Gemini API."

    st.write(generate_insights(df, f'{x_variable} and {y_variable}', 'Cluster', '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'))
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
    return silhouette,davies_bouldin,calinski_harabasz


##DBScan
def dbscan(df, best_eps, best_min_samples):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    df['DBCluster'] = dbscan.fit_predict(df)
    labels_dB = df['DBCluster'].values
    silhouette = metrics.silhouette_score(df, labels_dB)
    davies_bouldin = metrics.davies_bouldin_score(df, labels_dB)
    calinski_harabasz = metrics.calinski_harabasz_score(df,labels_dB)

    # Define plot configurations
    plot_config = [
        ('Schizophrenia', 'Depressive'),
        ('Depressive', 'Anxiety'),
        ('Anxiety', 'Bipolar'),
        ('Bipolar', 'Eating'),
        ('Eating', 'Schizophrenia')
    ]

    # Create subplots
    fig = make_subplots(rows=5, cols=2, subplot_titles=[
        'Schizophrenia vs Depressive Without Clustering', 'Schizophrenia vs Depressive With Clustering',
        'Depressive vs Anxiety Without Clustering', 'Depressive vs Anxiety With Clustering',
        'Anxiety vs Bipolar Without Clustering', 'Anxiety vs Bipolar With Clustering',
        'Bipolar vs Eating Without Clustering', 'Bipolar vs Eating With Clustering',
        'Eating vs Schizophrenia Without Clustering', 'Eating vs Schizophrenia With Clustering'
    ])

    # Plotting
    for i, (x, y) in enumerate(plot_config, start=1):
        # Without clustering
        fig.add_trace(
            go.Scatter(x=df[x], y=df[y], mode='markers', name=f'{x} vs {y} Without Clustering'),
            row=i, col=1
        )
        # With clustering
        fig.add_trace(
            go.Scatter(x=df[x], y=df[y], mode='markers', name=f'{x} vs {y} With Clustering',
                       marker=dict(color=labels_dB),
                       text=df['DBCluster'].apply(lambda label: f'Cluster: {label}')),
            row=i, col=2
        )

    # Update layout
    fig.update_layout(height=1500, width=1000, showlegend=False, title_text='DBSCAN Clustering Plots')

    # Show the figure
    st.plotly_chart(fig)
    return silhouette,davies_bouldin,calinski_harabasz


    

    
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
        silhouette,davies_bouldin,calinski_harabasz=kmeans(scaled_df,df,cluster_data,k)
        st.title("Performance Metrics")
        st.write(f"Silhouette Coefficient: {silhouette}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
        

    

    elif selected_option == "DBSCAN":
        eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=10, value=5, step=1)
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Epsilon (eps): {eps}")
        st.write(f"Minimum Samples: {min_samples}")
        scaled_df,df,cluster_data=preprocessing()
        silhouette,davies_bouldin,calinski_harabasz= dbscan(df, eps, min_samples)
        st.title("Performance Metrics")
        st.write(f"Silhouette Coefficient: {silhouette}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
        # Call function to run DBSCAN algorithm with selected parameters

    elif selected_option == "Hierarchical":
        st.write(f"Selected Algorithm: {selected_option}")
        scaled_data,df,cluster_data=preprocessing()
        silhouette,davies_bouldin,calinski_harabasz=hierarchical_clustering(df)
        # Display the selected option
        st.title("Performance Metrics")
        st.write(f"Silhouette Coefficient: {silhouette}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")



    with col2:
        st.write("Model Performance")
        st.write("Graphs")

    if st.button("Testing Button"):
        switch_page("page5")

