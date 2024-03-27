import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Mindful Marketing for Mental Wellness")
st.write("Our project focuses on how mental illnesses affect different regions of the country")
def save_uploaded_file(uploadedfile):
    with open(os.path.join("./", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("./", uploadedfile.name)

col1, col2 = st.columns(2)

with col1: 
    st.write("Project Details:")
    st.write("Project Details:")
with col2:
    st.write("This is column 2")

    st.write("Upload a CSV file:")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.success(f"File saved to: {file_path}")
        if st.button("Data Vsiualization"):
            switch_page("page2")





