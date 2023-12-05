import streamlit as st
from fastai.vision.all import *

st.title("🎅🏻 Xmas Bauble 🎄")

#image upload

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

#define label func
def label_func(fn): 
    return path/"labels_real_mask"/f"{fn.name[:-3]}png"

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    model = load_learner('resnet18_model.pkl')

    img.show()
    prediction = model.predict(img)
    prediction_mask = prediction[0]
    st.image(img)
    st.write(prediction_mask.shape)

