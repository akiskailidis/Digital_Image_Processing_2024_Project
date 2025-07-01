# Kailidis Kyrillos AM:4680

import os
import numpy as np
from common import read_img, save_img

def image_patches(image, patch_size=(16, 16)):
    """
    --- Zitima 1.a ---
    Given an input image and patch_size, return the corresponding image patches made
    by dividing up the image into patch_size sections and normalizing them.

    Input- image: H x W
           patch_size: a scalar tuple (M, N)
    Output- results: a list of images of size M x N
    """
    patches = []
    H, W = image.shape
    M, N = patch_size
    
    for i in range(0, H, M):
        for j in range(0, W, N):
            patch = image[i:i+M, j:j+N]
            if patch.shape == (M, N):  # Ensure patch is of correct size
                # Normalize patch to have zero mean and unit variance
                patch = (patch - np.mean(patch)) / np.std(patch)
                patches.append(patch)
    return patches

def convolve(image, kernel):
    """
    --- Zitima 2.b ---
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    H, W = image.shape
    h, w = kernel.shape
    pad_height = h // 2
    pad_width = w // 2

    # Pad image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image)

    # Flip the kernel for convolution
    kernel_flipped = np.flip(kernel)

    # Perform convolution
    for i in range(H):
        for j in range(W):
            region = padded_image[i:i+h, j:j+w]
            output[i, j] = np.sum(region * kernel_flipped)
    
    return output

def edge_detection(image):
    """
    --- Zitima 2.f ---
    Return Ix, Iy and the gradient magnitude of the input image.

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # Sobel kernels for edge detection
    kx = np.array([-1, 0, 1]).reshape((1, 3))  # 1 x 3
    ky = np.array([-1, 0, 1]).reshape((3, 1))  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # Gradient magnitude
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)
    
    return Ix, Iy, grad_magnitude

def sobel_operator(image):
    """
    --- Zitima 3.b ---
    Return Gx, Gy, and the gradient magnitude using the Sobel operator.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # Sobel filters
    Sx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])    # Horizontal
    Sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])  # Vertical

    Gx = convolve(image, Sx)
    Gy = convolve(image, Sy)

    # Gradient magnitude
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    return Gx, Gy, grad_magnitude

# Function to create a Gaussian kernel
def gaussian_kernel(size=3, standard_deviation=0.572):                           # 3x3 array with Standard Deviation = 0.572
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(standard_deviation))
    return kernel / np.sum(kernel)

def main():
    # The main function
    img = read_img('./grace_hopper.png')
    
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- Zitima 1: Image Patches --
    # (a)
    patches = image_patches(img)
    # Now choose any three patches and save them
    chosen_patches = np.concatenate(patches[:3], axis=1)            # Stack the first three patches horizontally
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- Zitima 2: Convolution and Gaussian Filter --
    # (c)
    # Calculate the Gaussian kernel described in the question 2.(c).
    kernel_gaussian = gaussian_kernel()                             # Calling gaussian_kernel() function
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")

    # -- Zitima 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- Zitima 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    print("LoG Filters are done.")

if __name__ == "__main__":
    main()
