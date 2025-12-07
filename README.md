# A Simple Vision System for the Blocks World

---

## Abstract
This project implements a classical computer-vision system for detecting and classifying geometric objects within the Blocks World environment. The system processes grayscale images, applies thresholding, extracts connected regions, and computes features such as area, aspect ratio, and compactness. A rule-based classifier uses these features to distinguish between circles, squares, rectangles, and general blocks. The pipeline provides an interpretable, deterministic, and lightweight vision solution for analyzing Blocks World images.

---

## Framework and Tools

- **Programming Language:** Python 3.x  
- **Libraries:**  
  - `opencv-python` – image processing and contour detection  
  - `numpy` – numerical computations and feature storage  
  - `matplotlib` – visualization of intermediate outputs  

---

## System Overview

- Grayscale conversion and image normalization  
- Thresholding using Otsu’s method  
- Connected-component labeling using DFS  
- Feature extraction:  
  - Area  
  - Perimeter  
  - Bounding box dimensions  
  - Aspect ratio  
  - Compactness  
  - Centroid  
- Rule-based classification based on feature similarity  
- Training performed on images 1–6  
- Testing performed on images 7–10  
- Similarity scoring using area, aspect ratio, and compactness  

---

## References

- **Textbook:** Chapter 2 – *A Simple Vision System for the Blocks World*  
- **OpenCV Documentation:** https://docs.opencv.org  
- **NumPy Documentation:** https://numpy.org/doc/  

---
