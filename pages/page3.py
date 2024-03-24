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
    excel_file = pd.ExcelFile("mentalall.xlsx") 
    dfs = pd.read_excel(excel_file, sheet_name=['1', '4','6','7']) 

    import streamlit as st
    import pandas as pd
    import plotly.express as px

    class Color:
        BLUE = "<span style='color: blue;'>"
        BOLD = "<span style='font-weight: bold;'>"
        UNDERLINE = "<span style='text-decoration: underline;'>"
        END = "</span>"

# Read data from the desired sheet
df = excel_file.parse("4")
# Sort DataFrame by "Major depression"
df.sort_values(by="Major depression", inplace=True)
# Define Streamlit app
st.title("Visualization of Major Depression")
# Create a bar chart using Plotly Express
fig = px.bar(df, x="Major depression", y="Entity", orientation='h', color='Bipolar disorder')
# Display the chart
st.plotly_chart(fig)


df.sort_values(by="Eating disorders", inplace=True)
# Define Streamlit app
st.title("Visualization of Eating DIsorder")
# Create a bar chart using Plotly Express
fig = px.bar(df, x="Eating disorders", y="Entity", orientation='h', color='Dysthymia')
# Display the chart
st.plotly_chart(fig)

df2 = excel_file.parse("4")
# Replace "<0.1" with 0.1 and convert column to float
df2.replace(to_replace="<0.1", value=0.1, regex=True, inplace=True)
df2['Schizophrenia'] = df2['Schizophrenia'].astype(float)
# Sort DataFrame by "Schizophrenia"
df2.sort_values(by="Schizophrenia", inplace=True)
# Define Streamlit app
st.title("Visualization of Schizophrenia")
# Create a bar chart using Plotly Express
fig = px.bar(df2, x="Schizophrenia", y="Entity", orientation='h', color='Anxiety disorders')
# Display the chart
st.plotly_chart(fig)

with col2:
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

# Load the Excel file
xl_file = pd.ExcelFile("mentalall.xlsx")

# Read data from the desired sheet
df2 = xl_file.parse("4")

# Create a new Streamlit app
st.title("Mental Health Visualization")

# Make subplots
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

x1 = ["Andean Latin America", "West Sub-Saharan Africa", "Tropical Latin America", "Central Asia", "Central Europe",
    "Central Sub-Saharan Africa", "Southern Latin America", "North Africa/Middle East", "Southern Sub-Saharan Africa",
    "Southeast Asia", "Oceania", "Central Latin America", "Eastern Europe", "South Asia", "East Sub-Saharan Africa",
    "Western Europe", "World", "East Asia", "Caribbean", "Asia Pacific", "Australasia", "North America"]

# Append bar plot to subplots
fig.append_trace(go.Bar(
    x=df2["Bipolar disorder"],
    y=x1,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(20, 10, 56, 1.0)',
            width=0),
    ),
    name='Bipolar disorder in Mental Health',
    orientation='h',
), 1, 1)

# Append scatter plot to subplots
fig.append_trace(go.Scatter(
    x=df2["Major depression"], y=x1,
    mode='lines+markers',
    line_color='rgb(40, 0, 128)',
    name='Major depression in Mental Health',
), 1, 2)

# Update layout
fig.update_layout(
    title='Major depression and Bipolar disorder',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=5,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=10000,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

# Adding labels
for ydn, yd, xd in zip(df2["Major depression"], df2["Bipolar disorder"], x1):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn+10,
                            text='{:,}'.format(ydn) + '%',
                            font=dict(family='Arial', size=10,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd+10,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=10,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))
# Source
annotations.append(dict(xref='paper', yref='paper',
                        x=-0.2, y=-0.109,
                        text="Mental health visualization",
                        font=dict(family='Arial', size=20, color='rgb(150,150,150)'),
                        showarrow=False))

fig.update_layout(annotations=annotations)
# Display the chart
st.plotly_chart(fig)


if st.button("Modelling Button"):
    switch_page("page4")