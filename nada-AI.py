#  streamlit run nada-AI.py
import numpy as np
import streamlit as st
import torch
from PIL import Image



model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_objects(image):
    image_np = np.array(image)
    results = model(image_np)
    return results

st.title("Slash AI intern - nada abodegham")
upload = st.file_uploader("upload image", type=["jpg", "jpeg", "png"])

if st.button("Analyse Image :)"):
    if upload:
        image = Image.open(upload)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Analyzing.... :)")
        
        results = detect_objects(image)
        st.image(results.render()[0], caption="Detected Objects", use_column_width=True)
        
        st.write("Detected components in image:")
        for i, row in results.pandas().xyxy[0].iterrows():
            label = row['name']
            st.write(f"{i+1}: {label}")

