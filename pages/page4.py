import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.title("Data Modelling")

col1, col2 = st.columns(2)

with col1: 

    st.header("Select one from below")

    # Define options for the dropdown
    options = ["K-means", "Clustering", "DBscan"]

    # Create the dropdown widget
    selected_option = st.selectbox("Select an option:", options)

    if selected_option == "KMeans":
        k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=3, step=1)
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Number of Clusters (k): {k}")
        # Call function to run KMeans algorithm with selected parameters

    elif selected_option == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.sidebar.slider("Minimum Samples", min_value=1, max_value=10, value=5, step=1)
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Epsilon (eps): {eps}")
        st.write(f"Minimum Samples: {min_samples}")
        # Call function to run DBSCAN algorithm with selected parameters

    elif selected_option == "Hierarchical":
        linkage = st.sidebar.selectbox("Linkage Method", ["single", "complete", "average"])
        st.write(f"Selected Algorithm: {selected_option}")
        st.write(f"Linkage Method: {linkage}")

    # Display the selected option
    st.write("You selected:", selected_option)

    progress_bar = st.progress(0)

    # Simulate a task that takes time
for percent_complete in range(101):
        time.sleep(0.05)  # Simulate a delay
        progress_bar.progress(percent_complete)

with col2:
    st.write("Model Performance")
    st.write("Graphs")

if st.button("Testing Button"):
    switch_page("page5")

