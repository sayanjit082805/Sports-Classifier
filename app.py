import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import random

st.set_page_config(
    page_title="Sports Classifier",
    page_icon="⚽",
    initial_sidebar_state="expanded",
)

sport_samples = [
    {
        "name": "Formula 1",
        "image": "public/f1.jpg",
    },
    {
        "name": "Cricket",
        "image": "public/cricket.jpg",
    },
    {
        "name": "Boxing",
        "image": "public/boxing.jpg",
    },
    {
        "name": "Football",
        "image": "public/football.jpg",
    },
    {
        "name": "Fencing",
        "image": "public/fencing.jpg",
    },
    {
        "name": "Nascar",
        "image": "public/nascar.jpg",
    },
    {
        "name": "Tennis",
        "image": "public/tennis.jpg",
    },
    {
        "name": "MotoGP",
        "image": "public/motogp.jpg",
    },
]

model = tf.keras.models.load_model("classifier.keras")
classes = joblib.load("class_list.pkl")

st.title("Sports Classifier")

st.subheader(
    "Classify different sports from images using the EfficientNet architecture.",
    divider="gray",
)

st.write(
    """
   A deep learning project to classify different sports into nearly 100 distinct types. The model is trained on the 100 Sports Image Classification dataset from Kaggle and uses the EfficientNet architecture for classification.     
"""
)

st.info(
    "Some classifications may be inaccurate. The model may not generalize perfectly to all images."
)

uploaded_file = st.file_uploader("Select an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Classify"):

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Preprocessing image...")
        progress_bar.progress(25)

        status_text.text("Running model inference...")
        progress_bar.progress(50)

        predictions = model.predict(img_array)
        progress_bar.progress(75)

        status_text.text("Processing results...")
        predicted_class = np.argmax(predictions, axis=1)[0]
        progress_bar.progress(100)

        status_text.text("Complete!")

        progress_bar.empty()
        status_text.empty()

        st.success(f"Prediction: **{classes[predicted_class].title()}**")

st.divider()

if "sport_samples" not in st.session_state:
    st.session_state.sport_samples = random.choice(sport_samples)

img = st.session_state.sport_samples
st.sidebar.image(img["image"], caption=img["name"])

st.sidebar.header("Dataset Information", divider="gray")

st.sidebar.markdown(
    """ 
    The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data). It contains around 36,000 images of aircraft of various different countries.

**Statistics** :

- **Total Images**: ~19,000 images
- **Training Set**: ~13,000 images
- **Test Set**: ~500 images
- **Validation Set**: ~500 images
- **Classes**: 100 distinct sports
"""
)

st.sidebar.header("Model Metrics", divider="gray")

st.sidebar.markdown(
    """ 
The model achieves an overall accuracy of ~94% on the validation data and ~97% on the test data.
"""
)

st.sidebar.markdown(
    """> The model is not intended for real-world applications and should not be used for any commercial or operational purposes."""
)