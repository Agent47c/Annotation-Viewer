import streamlit as st
import cv2
import os
import glob
import shutil
from PIL import Image
import numpy as np

st.set_page_config(layout="wide")
st.title("🔍 YOLO Annotation Viewer with File Move Option")

# --- Inputs ---
image_dir = st.text_input("🖼️ Image folder path", "D:/Done Dataset/Final Project_1/train/images")
label_dir = st.text_input("🏷️ Label folder path", "D:/Done Dataset/Final Project_1/train/labels")
class_file = st.text_input("📄 Class names file path", "D:/classes.txt")
move_dir = st.text_input("📁 Folder to move file (optional)", "D:/train/moved_files")

# --- Cache class names ---
@st.cache_data
def load_class_names(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        return []

class_names = load_class_names(class_file)
if not class_names:
    st.error(f"Class names file not found or empty at {class_file}")
    st.stop()

# --- Cache image list once ---
@st.cache_data
def get_image_files(img_dir):
    return sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")))

image_files = get_image_files(image_dir)
if not image_files:
    st.warning("No image files found.")
    st.stop()

# --- Session state for current index ---
if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0

# --- Image and label renderer (fast, no cache) ---
def draw_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_center, y_center, box_w, box_h = map(float, parts)
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                label = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"class_{int(cls_id)}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2, cv2.LINE_AA)
    return img

# --- Show current image with boxes ---
current_img = image_files[st.session_state.img_idx]
current_img_name = os.path.basename(current_img)
label_file = os.path.join(label_dir, current_img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

annotated_img = draw_boxes(current_img, label_file)
st.image(annotated_img, caption=current_img_name, width=1000)

# --- Navigation and Actions ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("⬅️ Previous"):
        st.session_state.img_idx = (st.session_state.img_idx - 1) % len(image_files)
with col2:
    if st.button("➡️ Next"):
        st.session_state.img_idx = (st.session_state.img_idx + 1) % len(image_files)
with col3:
    if st.button("📂 Move This File"):
        os.makedirs(move_dir, exist_ok=True)

        new_image_path = os.path.join(move_dir, current_img_name)
        new_label_path = os.path.join(move_dir, os.path.basename(label_file))

        shutil.move(current_img, new_image_path)
        if os.path.exists(label_file):
            shutil.move(label_file, new_label_path)

        st.success(f"✅ Moved {current_img_name} and label.")

        # Force reload image list and reset index
        image_files = get_image_files(image_dir)
        st.cache_data.clear()  # Clear cached image list
        st.rerun()
