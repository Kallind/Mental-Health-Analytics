import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Function to load and clean the data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Entity', 'Code', 'Year', 'Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)
    return df

# Function to create heatmap based on selected disorder
def create_heatmap(df, disorder):
    missing_vals_per_country = df[disorder].isna().groupby(df['Entity']).sum().reset_index(name='MissingCount')
    top_missing_countries = missing_vals_per_country.sort_values('MissingCount', ascending=False).head(5)['Entity']
    filtered_df = df[df['Entity'].isin(top_missing_countries)]
    filtered_df[f'{disorder} Available'] = filtered_df[disorder].notna().astype(int)
    heatmap_data = filtered_df.pivot_table(index='Entity', columns='Year', values=f'{disorder} Available', fill_value=0)
    return heatmap_data

# Function to plot yearly missing values across all disorders
def plot_missing_values(df):
    missing_values_by_year = df.isna().groupby(df['Year']).sum().sum(axis=1).reset_index(name='Total Missing Values')
    missing_values_normalized = (missing_values_by_year['Total Missing Values'] - missing_values_by_year['Total Missing Values'].min()) / (missing_values_by_year['Total Missing Values'].max() - missing_values_by_year['Total Missing Values'].min())
    color_scale = px.colors.sequential.Magenta
    colors = [color_scale[int(value * (len(color_scale) - 1))] for value in missing_values_normalized]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=missing_values_by_year['Year'],
        y=missing_values_by_year['Total Missing Values'],
        marker_color=colors,
        text=missing_values_by_year['Total Missing Values'],
        textposition='outside',
    ))
    fig.update_layout(
        title_text='Yearly Missing Values Across All Disorders',
        xaxis=dict(title='Year', showgrid=False),
        yaxis=dict(title='Total Missing Values', showgrid=True),
        plot_bgcolor='white',
        font=dict(size=12, color='black'),
    )
    return fig

# Function to detect outliers and plot prevalence data
def plot_outliers(df, disorder):
    df_disorder = df[['Entity', 'Year', disorder]].dropna()
    mean_val = df_disorder[disorder].mean()
    std_val = df_disorder[disorder].std()
    outliers_condition = np.abs(df_disorder[disorder] - mean_val) > 2 * std_val
    outliers = df_disorder[outliers_condition]
    normal_data = df_disorder[~outliers_condition]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal_data['Year'], y=normal_data[disorder], mode='markers',
                             marker=dict(color='lightblue', size=5), name='Normal'))
    fig.add_trace(go.Scatter(x=outliers['Year'], y=outliers[disorder], mode='markers',
                             marker=dict(color='red', size=8), name='Outliers',
                             text=outliers['Entity']))
    fig.add_trace(go.Scatter(x=df_disorder['Year'],
                             y=np.poly1d(np.polyfit(df_disorder['Year'], df_disorder[disorder], 1))(df_disorder['Year']),
                             mode='lines', name='Median', line=dict(color='grey', dash='dash')))
    fig.update_layout(
        title=f'{disorder} Prevalence: Normal Data Points vs. Outliers',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Prevalence'),
        plot_bgcolor='white',
        font=dict(size=12, color='black'),
    )
    return fig

# Load and clean data
file_path = 'asli_tatti.csv'
df = load_and_clean_data(file_path)

# Main Page
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.header("Data Visualisation without Data Cleaning")
st.write("This is the second page. Click the button below to go back to the main page.")

col1, col2 = st.columns(2)

with col1:
    st.write("Data")
    st.write("Graphs")

with col2:
    st.write("Graphs")
    st.write("Graphs")

if st.button("Clean Data and Visualize Data"):
    switch_page("page3")

# Page 3
st.page("page3")

st.subheader("Data Analysis Dashboard")

analysis_type = st.sidebar.selectbox('Select Analysis Type', ['Heatmap', 'Missing Values', 'Outliers'])

if analysis_type == 'Heatmap':
    st.subheader('Data Availability Heatmap for Selected Disorder')
    selected_disorder = st.selectbox('Select Disorder', df.columns[3:])
    heatmap_data = create_heatmap(df, selected_disorder)

    # Plot heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'
    ))
    fig.update_layout(
        xaxis=dict(title='Year', side="top"),
        yaxis=dict(title='Entity'),
        width=1000,
        height=550,
    )
    st.plotly_chart(fig)

elif analysis_type == 'Missing Values':
    st.subheader('Yearly Missing Values Across All Disorders')
    fig = plot_missing_values(df)
    st.plotly_chart(fig)

elif analysis_type == 'Outliers':
    st.subheader('Prevalence Data and Outliers')
    selected_disorder = st.selectbox('Select Disorder', df.columns[3:])
    fig = plot_outliers(df, selected_disorder)
    st.plotly_chart(fig)
