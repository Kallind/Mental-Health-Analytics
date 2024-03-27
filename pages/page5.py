import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import streamlit as st
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Model Comparison")
st.write("Comapring the performance of different clustering models namely K-means, Agglomerative and DBSCAN using Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Index.")

def save_uploaded_file(uploadedfile):
    with open(os.path.join("./", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("./", uploadedfile.name)



# def clustering_metrics_comparison():
#     data = {
#         'Model': ['K-means', 'Agglomerative', 'DBSCAN'],
#         'Silhouette Score': [0.5524, 0.4653, 0.4603],
#         'Davies-Bouldin Index': [0.7090, 0.8776, 0.6741],
#         'Calinski-Harabasz Index': [7882.5344, 5646.7036, 3107.1491]
#     }

#     metrics_df = pd.DataFrame(data)

#     # Bar Chart
#     fig_bar = make_subplots(rows=1, cols=3, subplot_titles=('Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'))

#     for i, metric in enumerate(metrics_df.columns[1:]):
#         fig_bar.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df[metric], name=metric), row=1, col=i+1)

#     fig_bar.update_layout(barmode='group', title_text='Clustering Models Comparison')

#     # Radar Chart
#     fig_radar = px.line_polar(metrics_df, r=['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
#                               theta=['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
#                               color='Model', line_close=True, title='Radar Chart of Clustering Models Performance')

#     fig_radar.update_traces(fill='toself')

#     # Display the plots
#     st.plotly_chart(fig_bar)
#     st.plotly_chart(fig_radar)

# # Call the function to display the plots in the Streamlit app
# clustering_metrics_comparison()


import streamlit as st
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def clustering_metrics_comparison():
    data = {
        'Model': ['K-means', 'Agglomerative', 'DBSCAN'],
        'Silhouette Score': [0.5524, 0.4653, 0.4603],
        'Davies-Bouldin Index': [0.7090, 0.8776, 0.6741],
        'Calinski-Harabasz Index': [7882.5344, 5646.7036, 3107.1491]
    }

    metrics_df = pd.DataFrame(data)

    # Bar Chart
    fig_bar = make_subplots(rows=1, cols=3, subplot_titles=('Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'))

    for i, metric in enumerate(metrics_df.columns[1:]):
        fig_bar.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df[metric], name=metric), row=1, col=i+1)

    fig_bar.update_layout(barmode='group', title_text='Clustering Models Comparison')

    # Radar Chart
    fig_radar = px.line_polar(metrics_df, r=['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
                              theta=['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
                              color='Model', line_close=True, title='Radar Chart of Clustering Models Performance')

    fig_radar.update_traces(fill='toself')

    # Display the plots with bigger size and center alignment
    st.plotly_chart(fig_bar, use_container_width=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.write(""" 
    **Conclusion** 
     Considering the information from both the bar and radar charts, we can conclude:
 
1. **K-means** stands out as the top performer:
   - It achieves the highest silhouette score, indicating well-separated clusters.
   - While its Davies-Bouldin Index is not the lowest, it's moderate, suggesting a reasonable balance of cluster tightness and separation.
   - The Calinski-Harabasz Index is substantially higher for K-means compared to the other models, indicating that clusters are very well-defined and distinct.
 
2. **Agglomerative clustering** shows a varied performance:
   - Its silhouette score is the second highest, which is commendable, but not as high as K-means, suggesting that clusters may not be as distinct.
   - It scores the worst on the Davies-Bouldin Index, which could indicate that its clusters are less compact or less well separated than those from K-means and DBSCAN.
   - The Calinski-Harabasz Index is also lower than K-means but higher than DBSCAN, placing it in an intermediate position for cluster definition.
 
3. **DBSCAN** demonstrates certain strengths despite a lower overall performance:
   - It has the lowest silhouette score, implying that the clusters may have some overlap or the points within clusters are not as tightly grouped.
   - The model does better in terms of the Davies-Bouldin Index than Agglomerative clustering, which suggests that its clusters, while not as tight as K-means, may be better separated than those of Agglomerative clustering.
   - DBSCAN has the lowest Calinski-Harabasz Index, suggesting its clusters may not be as well defined, which can be a characteristic of this method, especially with datasets having noise and outliers.
 
The radar chart specifically allows us to visually gauge the overall clustering performance profile of each method, where the 'area' covered by each model's 'slice' of the chart gives a sense of its effectiveness. K-means has the largest area, followed by Agglomerative clustering, with DBSCAN covering the least area. This visual comparison complements the bar chart by showing the relative performance of each clustering algorithm across multiple metrics simultaneously.
 
In summary, K-means tends to perform best according to these metrics, indicating it might be the most suitable model for this dataset when considering cluster separation and definition. However, DBSCAN shows reasonable performance on the Davies-Bouldin Index, hinting that with the right parameters, it could yield well-separated clusters. The choice of model should also consider the specific characteristics of the dataset and the requirements of the clustering task.""")

# Call the function to display the plots in the Streamlit app
clustering_metrics_comparison()


        


  

