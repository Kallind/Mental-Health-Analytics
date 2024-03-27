import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import streamlit as st
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Page 6")
st.write("This is the main page. Click the button below to go to the second page.")
def save_uploaded_file(uploadedfile):
    with open(os.path.join("./", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("./", uploadedfile.name)

col1, col2 = st.columns(2)


if 'labels' in st.session_state:
  label = st.session_state['labels']
else:
  st.warning('MKC')


def display_performance_metrics(df, label):
    silhouette = silhouette_score(df, label)
    davies_bouldin = davies_bouldin_score(df, label)
    calinski_harabasz = calinski_harabasz_score(df, label)

    st.title("Performance Metrics")
    st.write(f"Silhouette Coefficient: {silhouette}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")

    st.title("Performance Metrics")
    st.write("This is column 1")

    
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
  

    



  with col2:
    st.write("This is column 2")

    st.button("Performance matrix")
            
        

