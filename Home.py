import streamlit as st
from PIL import Image


# Add a title
st.set_page_config(page_title="Select Diagnosis", layout="centered")



st.title("Medical Diagnosis App")

st.markdown("")
st.markdown("<li> Currently Brain Tumors , Xrays and Skin Leison Analysis are ready for diagnosis </li>"
            "<li>The Models also explain what area in the images is the cause of diagnosis </li>"
            "<li>Currently the models are trained on a small dataset and will be trained on a larger dataset in the future</li>"
            '<li> The Application also provides generated information on how to diagnose the disease and what should the patient do in that case</li>'
            ,unsafe_allow_html=True)

with st.sidebar.container():
    image = Image.open("/Users/vikram/Downloads/Meditechlogo.png")
    st.image(image, caption='Meditech',use_column_width=True)




