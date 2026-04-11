import io
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Smile Design Comparison", layout="wide")

LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
UPPER_FACE_LANDMARKS_FOR_MIDLINE = [1, 168, 6, 197, 195, 5, 4]


@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to_same(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return (
        cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA),
        cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA),
    )


def extract_landmarks(image_bgr: np.ndarray) -> np.ndarray:
    result = get_face_mesh().process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        raise ValueError("No face detected in one of the uploaded images.")
    h, w = image_bgr.shape[:2]
    points = np.array(
        [(lm.x * w, lm.y * h) for lm in result.multi_face_landmarks[0].landmark],
        dtype=np.float32,
    )
    return points


def align_by_mouth_corners(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    left = landmarks[LEFT_MOUTH_CORNER]
    right = landmarks[RIGHT_MOUTH_CORNER]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    angle = np.degrees(np.arctan2(dy, dx))
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.mean(diff ** 2))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(ssim(gray1, gray2, data_range=255))


def mouth_metrics(image: np.ndarray, landmarks: np.ndarray) -> dict:
    left = landmarks[LEFT_MOUTH_CORNER]
    right = landmarks[RIGHT_MOUTH_CORNER]
    upper = landmarks[UPPER_LIP_CENTER]
    lower = landmarks[LOWER_LIP_CENTER]

    intercomm = float(np.linalg.norm(right - left))
    facial_midline_x = float(np.mean(landmarks[UPPER_FACE_LANDMARKS_FOR_MIDLINE, 0]))
    dental_midline_x = float((upper[0] + lower[0]) / 2.0)
    midline_dev = abs(dental_midline_x - facial_midline_x)

    cx = (left[0] + right[0]) / 2.0
    cy = (upper[1] + lower[1]) / 2.0
    bw = int(intercomm * 1.25)
    bh = int(intercomm * 0.75)
    x1 = max(int(cx - bw / 2), 0)
    y1 = max(int(cy - bh / 2), 0)
    x2 = min(int(cx + bw / 2), image.shape[1] - 1)
    y2 = min(int(cy + bh / 2), image.shape[0] - 1)

    return {
        "intercommissure_px": intercomm,
        "smile_width_px": intercomm,
        "facial_midline_x": facial_midline_x,
        "dental_midline_x": dental_midline_x,
        "midline_deviation_px": float(midline_dev),
        "mouth_box": (x1, y1, x2, y2),
        "landmarks": landmarks,
    }


def extract_mouth_roi(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def extract_incisal_curve(mouth_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 50, 130)
    h, w = edges.shape
    curve = np.full(w, np.nan, dtype=np.float32)
    for x in range(w):
        ys = np.where(edges[:, x] > 0)[0]
        if len(ys) > 0:
            curve[x] = float(np.min(ys))
    return curve


def compare_curves(curve1: np.ndarray, curve2: np.ndarray) -> float:
    valid = ~np.isnan(curve1) & ~np.isnan(curve2)
    if valid.sum() < 20:
        return float("nan")
    return float(np.mean(np.abs(curve1[valid] - curve2[valid])))


def draw_midlines(image: np.ndarray, metrics: dict, label: str) -> np.ndarray:
    vis = image.copy()
    x_face = int(metrics["facial_midline_x"])
    x_dental = int(metrics["dental_midline_x"])
    h = vis.shape[0]
    cv2.line(vis, (x_face, 0), (x_face, h), (255, 0, 0), 2)
    cv2.line(vis, (x_dental, 0), (x_dental, h), (0, 0, 255), 2)
    x1, y1, x2, y2 = metrics["mouth_box"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(vis, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    return vis


def make_overlay(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img1, img2 = resize_to_same(img1, img2)
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def compare_images(ai_img: np.ndarray, exo_img: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    lm_ai = extract_landmarks(ai_img)
    lm_exo = extract_landmarks(exo_img)

    ai_aligned = align_by_mouth_corners(ai_img, lm_ai)
    exo_aligned = align_by_mouth_corners(exo_img, lm_exo)
    ai_aligned, exo_aligned = resize_to_same(ai_aligned, exo_aligned)

    lm_ai2 = extract_landmarks(ai_aligned)
    lm_exo2 = extract_landmarks(exo_aligned)
    met_ai = mouth_metrics(ai_aligned, lm_ai2)
    met_exo = mouth_metrics(exo_aligned, lm_exo2)

    ssim_score = compute_ssim(ai_aligned, exo_aligned)
    mse_score = compute_mse(ai_aligned, exo_aligned)

    intercomm_mean = (met_ai["intercommissure_px"] + met_exo["intercommissure_px"]) / 2.0
    midline_diff_px = abs(met_ai["dental_midline_x"] - met_exo["dental_midline_x"])
    smile_width_diff_px = abs(met_ai["smile_width_px"] - met_exo["smile_width_px"])

    ai_mouth = extract_mouth_roi(ai_aligned, met_ai["mouth_box"])
    exo_mouth = extract_mouth_roi(exo_aligned, met_exo["mouth_box"])
    ai_mouth, exo_mouth = resize_to_same(ai_mouth, exo_mouth)
    ai_curve = extract_incisal_curve(ai_mouth)
    exo_curve = extract_incisal_curve(exo_mouth)
    curve_mad_px = compare_curves(ai_curve, exo_curve)

    results = {
        "SSIM": round(ssim_score, 4),
        "MSE": round(mse_score, 2),
        "Midline difference (px)": round(float(midline_diff_px), 2),
        "Midline difference (normalized)": round(float(midline_diff_px / intercomm_mean), 4),
        "Smile width difference (px)": round(float(smile_width_diff_px), 2),
        "Smile width difference (normalized)": round(float(smile_width_diff_px / intercomm_mean), 4),
        "Incisal curve MAD (px)": None if np.isnan(curve_mad_px) else round(float(curve_mad_px), 2),
        "Incisal curve MAD (normalized)": None if np.isnan(curve_mad_px) else round(float(curve_mad_px / intercomm_mean), 4),
        "Reference intercommissure (px)": round(float(intercomm_mean), 2),
    }

    ai_vis = draw_midlines(ai_aligned, met_ai, "Image A")
    exo_vis = draw_midlines(exo_aligned, met_exo, "Image B")
    overlay = make_overlay(ai_vis, exo_vis)
    return results, ai_vis, exo_vis, overlay


def dataframe_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("AI Smile Design vs Conventional Design Comparison")
st.write(
    "Upload 2 frontal smile images from the same case, then compare them using image similarity, "
    "midline deviation, smile width difference, and incisal curve deviation."
)

with st.sidebar:
    st.header("How to use")
    st.write(
        "Use this app for paired images from the same patient and same base view. "
        "Best results happen when both images have similar crop, head position, and lighting."
    )
    st.info(
        "Image A can be your AI simulation and Image B can be your Exocad or conventional design."
    )

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"], key="a")
with col2:
    file_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"], key="b")

label_a = st.text_input("Label for Image A", value="AI Simulation")
label_b = st.text_input("Label for Image B", value="Conventional Design")

run = st.button("Compare images", type="primary")

if run:
    if file_a is None or file_b is None:
        st.error("Please upload both images first.")
    else:
        try:
            img_a = pil_to_bgr(Image.open(file_a))
            img_b = pil_to_bgr(Image.open(file_b))

            results, vis_a, vis_b, overlay = compare_images(img_a, img_b)

            display_cols = st.columns(3)
            with display_cols[0]:
                st.image(bgr_to_rgb(vis_a), caption=label_a, use_container_width=True)
            with display_cols[1]:
                st.image(bgr_to_rgb(vis_b), caption=label_b, use_container_width=True)
            with display_cols[2]:
                st.image(bgr_to_rgb(overlay), caption="Overlay comparison", use_container_width=True)

            st.subheader("Comparison results")
            df = pd.DataFrame([results])
            st.dataframe(df, use_container_width=True)

            metric_cols = st.columns(4)
            metric_cols[0].metric("SSIM", results["SSIM"])
            metric_cols[1].metric("MSE", results["MSE"])
            metric_cols[2].metric("Midline diff (px)", results["Midline difference (px)"])
            metric_cols[3].metric("Curve MAD (px)", results["Incisal curve MAD (px)"])

            st.caption(
                "Blue line = facial midline proxy. Red line = dental midline proxy. "
                "Yellow box = mouth ROI used for curve extraction."
            )

            st.download_button(
                "Download results CSV",
                data=dataframe_download(df),
                file_name="smile_comparison_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Comparison failed: {e}")

st.markdown("---")
st.write(
    "Interpretation: higher SSIM suggests greater overall visual similarity, lower MSE suggests fewer pixel-level differences, "
    "and lower midline and curve deviation values suggest closer geometric agreement."
)
