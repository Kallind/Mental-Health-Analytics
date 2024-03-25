import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
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

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Data Visualisation with Data Cleaning")
st.write("This is the third page. Click the button below to go back to the main page.")

col1, col2 = st.columns(2)

with col1:
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go

def main():
    # Load data
    df = pd.read_csv('mental_cleaned.csv')
    df.columns = ['Country', 'Code', 'Year', 'Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']
    # Streamlit App
    st.title("Global Mental Health Disorders Prevalence")
    # Select Year
    year_of_interest = st.slider("Select Year", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=2010)
    # Filter data for selected year
    data_filtered = df[df['Year'] == year_of_interest]
    # List of disorders to include in the dropdown
    disorders = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']
    # Initialize figure
    fig = go.Figure()
    # Add traces, one for each disorder
    for disorder in disorders:
        fig.add_trace(
            go.Choropleth(
                locations = data_filtered['Code'],
                z = data_filtered[disorder],
                text = data_filtered['Country'],
                colorscale = 'Viridis',
                autocolorscale=False,
                reversescale=True,
                marker_line_color='darkgray',
                marker_line_width=0.5,
                colorbar_tickprefix = '%',
                colorbar_title = 'Prevalence<br>Rate',
                visible = (disorder == disorders[0])  # Only the first disorder is visible initially
            )
        )
    # Make dropdowns
    buttons = []

    for i, disorder in enumerate(disorders):
        button = dict(
            label=disorder,
            method="update",
            args=[{"visible": [False] * len(disorders)},
                  {"title": f"Global Prevalence of {disorder} in {year_of_interest}"}])
        button["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        buttons.append(button)
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        title_text=f"Global Prevalence of {disorders[0]} in {year_of_interest}",
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )
    # Plotly chart in Streamlit
    st.plotly_chart(fig)
if __name__ == "__main__":
    main()


with col2:
    st.write("This is column 2")
    st.write("This is the data visualisation page")


if st.button("Modelling Button"):
    switch_page("page4")