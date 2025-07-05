# Digital Image Processing Filters

This project contains Python implementations of fundamental image processing operations, developed as part of a university assignment in the course "Digital Image Processing" during the Spring Semester 2023–2024.
 
## Contents

The assignment covers the following topics in image processing:

### 1. Image Patches
- Function: `image_patches()`
- Splits a grayscale image into non-overlapping 16x16 patches.
- Each patch is normalized to have zero mean and unit variance.
- Useful for similarity analysis and object recognition.

### 2. Gaussian Filtering
- Function: `convolve()`
- Demonstrates equivalence between 2D Gaussian convolution and two successive 1D convolutions.
- Applied a 3x3 Gaussian filter with standard deviation σ = 0.572.
- Emphasizes the importance of kernel sum equal to 1 for preserving image intensity.

### 3. Edge Detection
- Function: `edge_detection()`
- Computes gradient magnitude using horizontal and vertical derivatives.
- Compares edge maps before and after Gaussian smoothing to illustrate noise reduction.

### 4. Sobel Operator
- Function: `sobel_operator()`
- Applies horizontal (Sx) and vertical (Sy) Sobel filters.
- Computes directional gradients and overall edge magnitude.

### 5. Laplacian of Gaussian (LoG)
- Applies LoG1 and LoG2 filters.
- Compares their kernel structures and the effect on edge detection.
- Highlights differences in response intensity and detection behavior.

## Technologies

- Python
- NumPy
- Matplotlib
- PIL or OpenCV

## Project Structure

