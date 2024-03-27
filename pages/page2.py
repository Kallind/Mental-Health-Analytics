import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from streamlit_extras.switch_page_button import switch_page

# Assuming 'asli_tatti.csv' is accessible
FILE_PATH = 'asli_tatti.csv'
df = pd.read_csv(FILE_PATH)
df.columns = ['Entity', 'Code', 'Year', 'Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)

disorders = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorder', 'Eating disorders']

# Visualization 1: Heatmap for Data Availability
def create_heatmap(df, disorder):
    missing_vals_per_country = df[disorder].isna().groupby(df['Entity']).sum().reset_index(name='MissingCount')
    top_missing_countries = missing_vals_per_country.sort_values('MissingCount', ascending=False).head(5)['Entity']
    filtered_df = df[df['Entity'].isin(top_missing_countries)]
    filtered_df[f'{disorder} Available'] = filtered_df[disorder].notna().astype(int)
    heatmap_data = filtered_df.pivot_table(index='Entity', columns='Year', values=f'{disorder} Available', fill_value=0)
    return heatmap_data

# Visualization 2: Bar Chart for Missing Values by Year
def plot_missing_values_by_year(df):
    missing_values_by_year = df.isna().groupby(df['Year']).sum().sum(axis=1).reset_index(name='Total Missing Values')
    missing_values_normalized = (missing_values_by_year['Total Missing Values'] - missing_values_by_year['Total Missing Values'].min()) / (missing_values_by_year['Total Missing Values'].max() - missing_values_by_year['Total Missing Values'].min())
    color_scale = px.colors.sequential.Magenta
    colors = [color_scale[int(value * (len(color_scale) - 1))] for value in missing_values_normalized]
    fig = go.Figure(go.Bar(x=missing_values_by_year['Year'], y=missing_values_by_year['Total Missing Values'], marker_color=colors))
    fig.update_layout(title='Yearly Missing Values Across All Disorders', xaxis_title='Year', yaxis_title='Total Missing Values')
    return fig

# Streamlit App Layout
st.title("Mental Health Disorders Data Visualizations")

# Dropdown for selecting disorder for heatmap
selected_disorder = st.selectbox("Select a disorder for heatmap visualization:", disorders)
heatmap_data = create_heatmap(df, selected_disorder)
fig_heatmap = go.Figure(go.Heatmap(x=heatmap_data.columns, y=heatmap_data.index, z=heatmap_data.values, colorscale=px.colors.sequential.Mint))
fig_heatmap.update_layout(title=f"{selected_disorder} Data Availability", xaxis_title='Year', yaxis_title='Entity')
st.plotly_chart(fig_heatmap, use_container_width=True)

# Show bar chart for missing values by year
st.header("Missing Values by Year Across All Disorders")
fig_missing_values = plot_missing_values_by_year(df)
st.plotly_chart(fig_missing_values, use_container_width=True)

# Note: The third visualization requires adapting the provided code to include dynamic updates based on dropdown selections within Streamlit.
# This placeholder indicates where you would add the functionality.
st.header("Outliers Visualization")
# Implement the outliers visualization based on the selected disorder here
st.write("Outliers visualization functionality to be implemented.")

# Visualization 3: Outliers Visualization

# Include previously defined visualization functions here...

# Visualization 3: Outliers Detection and Visualization
def outliers_viz(disorder):
    df_disorder = df[['Entity', 'Year', disorder]].dropna()
    mean_val = df_disorder[disorder].mean()
    std_val = df_disorder[disorder].std()
    outliers_condition = np.abs(df_disorder[disorder] - mean_val) > 2 * std_val
    outliers = df_disorder[outliers_condition]
    normal_data = df_disorder[~outliers_condition]

    fig = go.Figure()

    # Scatter for normal data points
    fig.add_trace(go.Scatter(x=normal_data['Year'], y=normal_data[disorder], mode='markers',
                             marker=dict(color='lightblue', size=5), name='Normal'))

    # Scatter for outliers
    fig.add_trace(go.Scatter(x=outliers['Year'], y=outliers[disorder], mode='markers',
                             marker=dict(color='red', size=8), name='Outliers',
                             text=outliers['Entity']))  # Entity as hover text

    # Trend line
    fig.add_trace(go.Scatter(x=df_disorder['Year'],
                             y=np.poly1d(np.polyfit(df_disorder['Year'], df_disorder[disorder], 1))(df_disorder['Year']),
                             mode='lines', name='Median', line=dict(color='grey', dash='dash')))

    fig.update_layout(title=f"{disorder} Prevalence: Normal Data Points vs. Outliers",
                      xaxis_title='Year', yaxis_title=disorder)

    return fig

# Dropdown for selecting disorder for outliers visualization
selected_disorder_outliers = st.selectbox("Select a disorder for outliers visualization:", disorders, key="outliers_dropdown")
fig_outliers = outliers_viz(selected_disorder_outliers)
st.plotly_chart(fig_outliers, use_container_width=True)

if st.button("Clean Data and Visualize Data"):
    switch_page("page3")
