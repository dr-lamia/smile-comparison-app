import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Smile Design Comparison", layout="wide")

st.title("AI Smile Design vs Conventional Design Comparison")
st.write(
    "Upload 2 frontal smile images from the same case, then compare them using "
    "image similarity and visual overlay."
)

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def resize_to_same(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img1, img2

def compute_mse(img1, img2):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.mean(diff ** 2))

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(ssim(gray1, gray2, data_range=255))

def make_overlay(img1, img2):
    return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

col1, col2 = st.columns(2)

with col1:
    file_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"])

with col2:
    file_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"])

label_a = st.text_input("Label for Image A", value="AI Simulation")
label_b = st.text_input("Label for Image B", value="Conventional Design")

if st.button("Compare images"):
    if file_a is None or file_b is None:
        st.warning("Please upload both images first.")
    else:
        try:
            img_a = load_image(file_a)
            img_b = load_image(file_b)

            img_a, img_b = resize_to_same(img_a, img_b)

            ssim_score = compute_ssim(img_a, img_b)
            mse_score = compute_mse(img_a, img_b)
            overlay = make_overlay(img_a, img_b)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(to_rgb(img_a), caption=label_a, use_container_width=True)
            with c2:
                st.image(to_rgb(img_b), caption=label_b, use_container_width=True)
            with c3:
                st.image(to_rgb(overlay), caption="Overlay", use_container_width=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("SSIM", f"{ssim_score:.4f}")
            with m2:
                st.metric("MSE", f"{mse_score:.2f}")

        except Exception as e:
            st.error(f"Comparison failed: {e}")
