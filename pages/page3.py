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
st.markdown("<h1 style='text-align: center;'>Data Visualisation with Data Cleaning</h1>", unsafe_allow_html=True)




def main1():
    # Load data
    df = pd.read_csv('mental_cleaned.csv')
    df.columns = ['Country', 'Code', 'Year', 'Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']

    # Streamlit App
    st.markdown("<div style='text-align: center; font-size: 30px;'>Global Mental Health Disorders Prevalence</div>", unsafe_allow_html=True)

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
                x=0.5,  # Center align the dropdown
                xanchor="center",  # Anchor point for x-coordinate
                y=1.15,
                yanchor="top"
            ),
        ],
        title_text=f"Global Prevalence of {disorders[0]} in {year_of_interest}",
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        # Max width and center aligned
        width=900,
        margin=dict(l=0, r=0, t=50, b=0),  # Adjust margins to center align
    )

    # Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True) 

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

    st.markdown("<div style='text-align: center; font-size: 30px;'>Prevalence of Mental Health Disorders Over Time by Country</div>", unsafe_allow_html=True)

    # Add dropdown to the figure
    fig.update_layout(showlegend=True,
                      updatemenus=[{"buttons": buttons, "direction": "down", "active": 0, "showactive": True,
                                    "x": 0.5, "xanchor": "center", "y": 1.1, "yanchor": "top"}])  # Center align dropdown

    # Update layout to add titles and axis labels, and adjust width
    fig.update_layout(
                      xaxis_title='Year',
                      yaxis_title='Prevalence (%)',
                      width=900  # Adjust the width as per your requirement
                      )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


     
def main3():
    # Select variables of interest
    df= pd.read_csv('mental_cleaned.csv')
    df1_ent=df.drop(["Entity","Year"],axis=1)
    df=df1_ent.groupby('Code').mean()
    
# New column names
    new_column_names = {
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating'
    }

    # Rename columns
    df.rename(columns=new_column_names, inplace=True)


    df1_variables = df[["Schizophrenia", "Depressive", "Anxiety", "Bipolar", "Eating"]]
    
    # Calculate correlation matrix
    Corrmat = df1_variables.corr()

    # Plot heatmap
    st.write("### Correlation Matrix")
    plt.figure(figsize=(10, 5), dpi=200)
    xx= sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5)
    plt.title('Correlation Matrix')
    st.pyplot(xx.figure)


main1()
main2()
main3()

##BUTTON
# Enlarge and center-align the button using CSS styling
button_style = "<style>div[data-testid='stButton']>button {width: 200px !important; height: 50px !important; text-align: center !important; margin: auto !important;}</style>"
st.markdown(button_style, unsafe_allow_html=True)

if st.button("Modelling Button"):
    switch_page("page4")