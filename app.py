import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="AI Smile Evaluator", layout="wide")

st.title("AI-Generated Smile vs Conventional Smile Design Comparison")
st.write("Upload the AI-generated smile image and the conventional smile design image from the same case.")

# -----------------------------
# Helper functions
# -----------------------------
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_to_same(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img1_resized, img2_resized

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(ssim(gray1, gray2, data_range=255))

def compute_mse(img1, img2):
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))

def make_overlay(img1, img2):
    return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

def crop_region(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1_px = int(x1 * w / 100)
    x2_px = int(x2 * w / 100)
    y1_px = int(y1 * h / 100)
    y2_px = int(y2 * h / 100)
    return img[y1_px:y2_px, x1_px:x2_px]

def draw_box(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    boxed = img.copy()
    h, w = boxed.shape[:2]
    pt1 = (int(x1 * w / 100), int(y1 * h / 100))
    pt2 = (int(x2 * w / 100), int(y2 * h / 100))
    cv2.rectangle(boxed, pt1, pt2, color, thickness)
    return boxed

def align_images_ecc(img_ref, img_to_align):
    """
    Align img_to_align to img_ref using ECC-based affine registration.
    Returns aligned image and warp matrix.
    """
    ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    ref_gray = ref_gray.astype(np.float32) / 255.0
    align_gray = align_gray.astype(np.float32) / 255.0

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        5000,
        1e-7
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray,
            align_gray,
            warp_matrix,
            warp_mode,
            criteria
        )

        aligned = cv2.warpAffine(
            img_to_align,
            warp_matrix,
            (img_ref.shape[1], img_ref.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )

        return aligned, warp_matrix

    except cv2.error:
        return img_to_align, warp_matrix

# -----------------------------
# Upload section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    file_ai = st.file_uploader("Upload AI-generated smile image", type=["jpg", "jpeg", "png"])

with col2:
    file_conv = st.file_uploader("Upload conventional smile design image", type=["jpg", "jpeg", "png"])

# -----------------------------
# Smile-region selection
# -----------------------------
st.subheader("Smile-region selection")
st.write("Adjust the crop to isolate the mouth/smile region.")

x1 = st.slider("Left boundary (%)", 0, 100, 30)
x2 = st.slider("Right boundary (%)", 0, 100, 75)
y1 = st.slider("Upper boundary (%)", 0, 100, 45)
y2 = st.slider("Lower boundary (%)", 0, 100, 75)

# -----------------------------
# Preview before comparison
# -----------------------------
if file_ai is not None and file_conv is not None:
    preview_ai = load_image(file_ai)
    preview_conv = load_image(file_conv)

    preview_ai, preview_conv = resize_to_same(preview_ai, preview_conv)

    st.subheader("Image previews")
    p1, p2 = st.columns(2)
    with p1:
        st.image(
            to_rgb(draw_box(preview_ai, x1, y1, x2, y2)),
            caption="AI-generated smile preview with smile-region box",
            use_container_width=True
        )
    with p2:
        st.image(
            to_rgb(draw_box(preview_conv, x1, y1, x2, y2)),
            caption="Conventional smile preview with smile-region box",
            use_container_width=True
        )

# -----------------------------
# Comparison button
# -----------------------------
if st.button("Compare AI vs Conventional Design"):
    if file_ai is None or file_conv is None:
        st.warning("Please upload both images first.")
    else:
        # Load images
        img_ai = load_image(file_ai)
        img_conv = load_image(file_conv)

        # Resize to same dimensions
        img_ai, img_conv = resize_to_same(img_ai, img_conv)

        # Align conventional image to AI image
        img_conv_aligned, _ = align_images_ecc(img_ai, img_conv)

        # Full-face comparison
        full_ssim = compute_ssim(img_ai, img_conv_aligned)
        full_mse = compute_mse(img_ai, img_conv_aligned)
        full_overlay = make_overlay(img_ai, img_conv_aligned)

        st.subheader("Full-face comparison")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(to_rgb(img_ai), caption="AI-generated smile", use_container_width=True)
        with c2:
            st.image(to_rgb(img_conv_aligned), caption="Conventional smile design (aligned)", use_container_width=True)
        with c3:
            st.image(to_rgb(full_overlay), caption="Full-face overlay", use_container_width=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Full-face SSIM", f"{full_ssim:.4f}")
        with m2:
            st.metric("Full-face MSE", f"{full_mse:.2f}")

        # Smile-region cropping
        mouth_ai = crop_region(img_ai, x1, y1, x2, y2)
        mouth_conv = crop_region(img_conv_aligned, x1, y1, x2, y2)

        # Resize cropped regions to same dimensions
        mouth_ai, mouth_conv = resize_to_same(mouth_ai, mouth_conv)

        # Additional alignment within smile region
        mouth_conv_aligned, _ = align_images_ecc(mouth_ai, mouth_conv)

        # Smile-region comparison
        mouth_ssim = compute_ssim(mouth_ai, mouth_conv_aligned)
        mouth_mse = compute_mse(mouth_ai, mouth_conv_aligned)
        mouth_overlay = make_overlay(mouth_ai, mouth_conv_aligned)

        st.subheader("Smile-region comparison")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.image(to_rgb(mouth_ai), caption="AI smile region", use_container_width=True)
        with c5:
            st.image(to_rgb(mouth_conv_aligned), caption="Conventional smile region (aligned)", use_container_width=True)
        with c6:
            st.image(to_rgb(mouth_overlay), caption="Smile-region overlay", use_container_width=True)

        m3, m4 = st.columns(2)
        with m3:
            st.metric("Smile-region SSIM", f"{mouth_ssim:.4f}")
        with m4:
            st.metric("Smile-region MSE", f"{mouth_mse:.2f}")

        # Summary note
        st.info(
            "For the manuscript, report full-face SSIM/MSE as identity-preservation similarity "
            "and smile-region SSIM/MSE as dentolabial design similarity. "
            "Paired images were aligned using ECC-based affine registration before comparison."
        )
