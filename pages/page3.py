import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Data Visualisation with Data Cleaning")
st.write("This is the third page. Click the button below to go back to the main page.")

col1, col2 = st.columns(2)

with col1: 
    st.write("Data")
    st.write("Graphs")
with col2:
    st.write("Graphs")
    st.write("Graphs")

if st.button("Modelling Button"):
    switch_page("page4")