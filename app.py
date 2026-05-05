import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="AI vs Conventional Smile Comparison", layout="wide")

st.title("AI-Generated Smile vs Conventional Smile Design Comparison")
st.write("Upload the AI-generated smile and the conventional smile design from the same case.")

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def resize_to_same(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return (
        cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA),
        cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    )

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(ssim(gray1, gray2, data_range=255))

def compute_mse(img1, img2):
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))

def make_overlay(img1, img2):
    return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def crop_region(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = int(x1 * w / 100)
    x2 = int(x2 * w / 100)
    y1 = int(y1 * h / 100)
    y2 = int(y2 * h / 100)
    return img[y1:y2, x1:x2]

col1, col2 = st.columns(2)

with col1:
    file_ai = st.file_uploader("Upload AI-generated smile image", type=["jpg", "jpeg", "png"])

with col2:
    file_conv = st.file_uploader("Upload conventional smile design image", type=["jpg", "jpeg", "png"])

st.subheader("Smile-region selection")
st.write("Adjust the crop to isolate the mouth/smile region.")

x1 = st.slider("Left boundary (%)", 0, 100, 25)
x2 = st.slider("Right boundary (%)", 0, 100, 75)
y1 = st.slider("Upper boundary (%)", 0, 100, 45)
y2 = st.slider("Lower boundary (%)", 0, 100, 75)

if st.button("Compare AI vs Conventional Design"):
    if file_ai is None or file_conv is None:
        st.warning("Please upload both images first.")
    else:
        img_ai = load_image(file_ai)
        img_conv = load_image(file_conv)

        img_ai, img_conv = resize_to_same(img_ai, img_conv)

        full_ssim = compute_ssim(img_ai, img_conv)
        full_mse = compute_mse(img_ai, img_conv)
        full_overlay = make_overlay(img_ai, img_conv)

        mouth_ai = crop_region(img_ai, x1, y1, x2, y2)
        mouth_conv = crop_region(img_conv, x1, y1, x2, y2)

        mouth_ai, mouth_conv = resize_to_same(mouth_ai, mouth_conv)

        mouth_ssim = compute_ssim(mouth_ai, mouth_conv)
        mouth_mse = compute_mse(mouth_ai, mouth_conv)
        mouth_overlay = make_overlay(mouth_ai, mouth_conv)

        st.subheader("Full-face comparison")
        c1, c2, c3 = st.columns(3)
        c1.image(to_rgb(img_ai), caption="AI-generated smile", use_container_width=True)
        c2.image(to_rgb(img_conv), caption="Conventional smile design", use_container_width=True)
        c3.image(to_rgb(full_overlay), caption="Full-face overlay", use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Full-face SSIM", f"{full_ssim:.4f}")
        m2.metric("Full-face MSE", f"{full_mse:.2f}")

        st.subheader("Smile-region comparison")
        c4, c5, c6 = st.columns(3)
        c4.image(to_rgb(mouth_ai), caption="AI smile region", use_container_width=True)
        c5.image(to_rgb(mouth_conv), caption="Conventional smile region", use_container_width=True)
        c6.image(to_rgb(mouth_overlay), caption="Smile-region overlay", use_container_width=True)

        m3, m4 = st.columns(2)
        m3.metric("Smile-region SSIM", f"{mouth_ssim:.4f}")
        m4.metric("Smile-region MSE", f"{mouth_mse:.2f}")

        st.info(
            "For the manuscript, report full-face SSIM/MSE as identity-preservation similarity "
            "and smile-region SSIM/MSE as dentolabial design similarity."
        )
