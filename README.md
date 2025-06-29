﻿# 🔍 YOLO Annotation Viewer & File Mover

A simple Streamlit-based tool to **visualize YOLO annotations** and **move annotated files** for dataset cleanup and organization.

## 🧠 Overview

This app was built to streamline the process of reviewing and organizing labeled data for object detection projects. It helped manage over **150,000 images** and **40,000 label files** in my final year project on PPE detection using **YOLOv11 + DeepSORT**.

## ✨ Features

- 📂 View annotated images with bounding boxes (YOLO format)
- ⏪ Easy navigation through thousands of images
- 🏷️ Display class labels from a class file
- ✅ Move selected images and their labels to a new folder
- ⚡ Built with Streamlit, OpenCV, and PIL

## 🖥️ How It Works

1. Load image and label directories
2. Load class names file
3. Browse through images with YOLO annotations
4. Move incorrectly labeled or unwanted files with a single click

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/Agent47c/Annotation-Viewer.git
cd Annotation-Viewer
```
### 2. Install Dependencies
```bash
pip install streamlit opencv-python pillow numpy
```
### 3. Run the App
```bash
streamlit run Dataset_BBChecking.py
```

### 📁 Folder Structure
- images
- labels
- classes.txt
### 🛠️ Tech Stack
- Python
- Streamlit
- OpenCV
- PIL
- NumPy
## 📷 Screenshot

![Screenshot](/UI.png)


## 🧑‍💻 Author
**Hamza Ramzan**  
Freelance PC Builder & Machine Learning Enthusiast  
[LinkedIn](https://www.linkedin.com/in/hamza-ramzan-5516a7272/) • [GitHub](https://github.com/Agent47c)

---

## 🌟 License

This project is licensed under the MIT License. Feel free to use, modify, and share!
