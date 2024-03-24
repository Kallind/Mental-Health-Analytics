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


df2.sort_values(by= "Eating disorders" ,inplace=True)
plt.figure(dpi=200)
fig = px.bar(df2, x="Eating disorders", y="Entity", orientation='h',color='Dysthymia')
fig.show()

with col2:
    st.write("Graphs")
    st.write("Graphs")

if st.button("Modelling Button"):
    switch_page("page4")