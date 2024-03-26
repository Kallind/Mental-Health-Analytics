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

def main1():
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

####Heatmap
    df1_variables = df[['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']]

    # Calculate correlation matrix
    Corrmat = df1_variables.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5)




##LINE PLOT

def main2():
    # Load the dataset (adjust the path as necessary)
    data = pd.read_csv('mental_cleaned.csv')

    # Sample structure adjustment if necessary (ensure the dataset is loaded correctly)
    data.columns = ['Entity', 'Code', 'Year', 'Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']

    # Convert 'Year' column to numeric, handling errors by coercing to NaN
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

    # Drop rows where 'Year' is NaN
    data = data.dropna(subset=['Year'])

    # Convert 'Year' column to integer
    data['Year'] = data['Year'].astype(int)

    # Ensure numerical columns have appropriate data types
    numerical_columns = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']
    data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')

    # Sort the data by year
    data = data.sort_values('Year')
    
    # Create a list of all unique countries/entities in the dataset
    countries = data['Entity'].unique()

    # Initialize the figure
    fig = go.Figure()

    # Placeholder for the initial plot (optional: choose a default country or the first in the list)
    default_country = countries[0]
    default_data = data[data['Entity'] == default_country]

    # Add a line for each disorder for the default country
    for disorder in ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']:
        fig.add_trace(go.Scatter(x=default_data['Year'], y=default_data[disorder], mode='lines', name=disorder))

    # Create dropdown menus
    buttons = []

    for country in countries:
        country_data = data[data['Entity'] == country]
        country_buttons = dict(label=country,
                               method='update',
                               args=[{'y': [country_data[disorder].values for disorder in
                                            ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders',
                                             'Bipolar disorder', 'Eating disorders']],
                                      'x': [country_data['Year'].values] * 5,
                                      'labels': ['Schizophrenia disorders', 'Depressive disorders',
                                                 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']}])
        buttons.append(country_buttons)

    buttons.sort(key=lambda x: x['label'])  # Sort buttons alphabetically by country name

    # Add dropdown to the figure
    fig.update_layout(showlegend=True,
                      updatemenus=[{"buttons": buttons, "direction": "down", "active": 0, "showactive": True,
                                    "x": 0.1, "xanchor": "left", "y": 1.1, "yanchor": "top"}])

    # Update layout to add titles and axis labels
    fig.update_layout(title='Prevalence of Mental Health Disorders Over Time by Country',
                      xaxis_title='Year',
                      yaxis_title='Prevalence (%)')

    # Display the plot
    st.plotly_chart(fig)



with col1:
   main1()



with col2:
    main2()
    st.write("This is column 2")
    st.write("This is the data visualisation page")


if st.button("Modelling Button"):
    switch_page("page4")