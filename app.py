import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.title("Smile Design Comparison Tool")

def load_image(file):
    image = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def resize_images(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))

def compare(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(gray1, gray2)
    mse = np.mean((img1 - img2) ** 2)

    overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    return ssim_score, mse, overlay

col1, col2 = st.columns(2)

with col1:
    img_a_file = st.file_uploader("Upload AI Image")

with col2:
    img_b_file = st.file_uploader("Upload Conventional Image")

if st.button("Compare"):
    if img_a_file and img_b_file:
        img1 = load_image(img_a_file)
        img2 = load_image(img_b_file)

        img1, img2 = resize_images(img1, img2)

        ssim_score, mse_score, overlay = compare(img1, img2)

        st.image([img1[:,:,::-1], img2[:,:,::-1], overlay[:,:,::-1]],
                 caption=["Image A", "Image B", "Overlay"])

        st.metric("SSIM", round(ssim_score, 3))
        st.metric("MSE", round(mse_score, 2))
    else:
        st.warning("Upload both images first")
