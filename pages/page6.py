import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Page 6")
st.write("This is the main page. Click the button below to go to the second page.")
def save_uploaded_file(uploadedfile):
    with open(os.path.join("./", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("./", uploadedfile.name)

col1, col2 = st.columns(2)

with col1: 
    st.write("This is column 1")
with col2:
    st.write("This is column 2")

    st.button("Performance matrix")
            
        

