import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>Mindful Marketing for Mental Wellness</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 20px;'>Our project focuses on how mental illnesses affect different regions of the country</div>", unsafe_allow_html=True)

st.image('https://images.thequint.com/thequint%2F2023-01%2F3b080752-7789-458b-a101-756f2a8ca860%2Fhero_image__1_.jpg?auto=format%2Ccompress&fmt=webp&width=120&w=1200', use_column_width=True)
def save_uploaded_file(uploadedfile):
    with open(os.path.join("./", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("./", uploadedfile.name)

col1, col2 = st.columns(2)


with col1: 
    st.write("What is Mental Health?")
    st.write("Mental health is a state of mental well-being that enables people to cope with the stresses of life, realize their abilities, learn well and work well, and contribute to their community. It is an integral component of health and well-being that underpins our individual and collective abilities to make decisions, build relationships and shape the world we live in. Mental health is a basic human right. And it is crucial to personal, community and socio-economic development.")
    st.write("What is Clustering?")
    st.write("When you're trying to learn about something, say music, one approach might be to look for meaningful groups or collections. You might organize music by genre, while your friend might organize music by decade. How you choose to group items helps you to understand more about them as individual pieces of music. You might find that you have a deep affinity for punk rock and further break down the genre into different approaches or music from different locations. On the other hand, your friend might look at music from the 1980's and be able to understand how the music across genres at that time was influenced by the sociopolitical climate. In both cases, you and your friend have learned something interesting about music, even though you took different approaches. In machine learning too, we often group examples as a first step to understand a subject (data set) in a machine learning system. Grouping unlabeled examples is called clustering.")
    st.write("Why Clustering in Mental Health?")
    


with col2:
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.success(f"File saved to: {file_path}")
        if st.button("Data Vsiualization"):
            switch_page("page2")





