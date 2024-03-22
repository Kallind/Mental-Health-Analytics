import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Model testing and evaluation")
st.write("This is the fifth page. Click the button below to go back to the main page.")

def save_uploaded_file(uploadedfile):
    with open(os.path.join("./", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("./", uploadedfile.name)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"File saved to: {file_path}")
    if st.button("Data Visualization"):
        switch_page("page2")
        

with col2:
    st.write("Inferencing")
    st.write("Model Predictions")

