# A Simple Vision System for the Blocks World

---

## Abstract
This project implements a computer-vision system for detecting, segmenting, and reconstructing 3D geometric objects in the Blocks World environment. The system first preprocesses images by converting them to grayscale, applying Gaussian smoothing to reduce noise, and performing binary segmentation using Otsu’s thresholding. Edges are then detected using Sobel gradients and classified as vertical or horizontal based on their orientation. The system segments figure (blocks) and ground regions using HSV color analysis and identifies contact edges (where objects touch the ground) and occlusion edges (where objects overlap).  

Using these preprocessing results, the 3D reconstruction module builds a sparse linear system incorporating multiple constraints: ground constraints, contact edge constraints, vertical and horizontal edge constraints, and smoothness/planarity constraints. The system solves this overdetermined system using sparse least-squares (LSQR) to estimate depth (Z) values for each pixel. Full 3D world coordinates (X, Y, Z) are recovered using parallel projection equations, allowing accurate surface reconstruction of blocks in 3D space. The pipeline is fully interpretable, deterministic, and generates visualizations of both preprocessing results and 3D reconstructions for analysis and verification.

---

## Framework and Tools

- **Programming Language:** Python 3.x  
- **Libraries:**  
  - `opencv-python` – image processing, gradients, and thresholding  
  - `numpy` – numerical computations and array handling  
  - `scipy` – sparse linear algebra for solving least-squares constraints  
  - `matplotlib` – visualization of preprocessing results and 3D reconstruction  

---

## System Overview

- Grayscale conversion and Gaussian smoothing  
- Binary segmentation using Otsu’s method  
- Edge detection using Sobel gradients (gx, gy)  
- Edge strength and orientation computation  
- Edge classification: vertical vs. horizontal  
- Figure/ground segmentation using HSV thresholds  
- Detection of contact edges (figure → ground) and occlusion edges (ground → figure)  
- Optional feature extraction: area, perimeter, bounding box, aspect ratio, compactness, centroid  
- 3D reconstruction pipeline:  
  - Build sparse constraint system (ground, contact, vertical, horizontal, smoothness)  
  - Solve using LSQR to estimate Z coordinates  
  - Recover X and Y coordinates using projection equations  
- Visualization of preprocessing results and 3D reconstructed surfaces  

---

## References

- **Textbook:** Chapter 2 – *3D Reconstruction in Blocks World*  
- **OpenCV Documentation:** https://docs.opencv.org  
- **NumPy Documentation:** https://numpy.org/doc/  
- **SciPy Sparse Linear Algebra:** https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html  
