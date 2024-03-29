from PIL import Image
from transformers import pipeline
import streamlit as st

@st.cache_resource()
def load_model_pipelines(task, model_path):
    model = pipeline(task, model=model_path)
    return model


st.title("Photo Semantic Finder")

model_path = "./Models/models--Salesforce--blip-image-captioning-large/snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90"

model = load_model_pipelines("image-to-text", model_path)


uploaded_image = st.file_uploader("Choose a photo", type=["png", "jpg", "jpeg"])

if st.button("Generate Semantics", disabled=uploaded_image is None):
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, width=300)
        with col2:
            with st.spinner('Generating Semantics...'):
                pil_image = Image.open(uploaded_image)
                semantics = model(images=pil_image)[0]['generated_text']
                st.subheader(semantics)