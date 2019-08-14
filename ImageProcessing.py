import numpy as np

def gaussian_blur(image, size, sigma):
    norm = 1. / (2. * np.pi * sigma**2)
    kernel = gaussian_kernel(size, sigma)
    return sep_convolution(image, kernel, kernel.T)

def gaussian_kernel(size, sigma):
    return np.fromfunction(lambda x: np.exp(-((x-(size//2))**2) / (2. * sigma**2)), (size,))

def sobel_filter(image):
    kernel_pt1 = np.array([1,2,1])
    kernel_pt2 = np.array([1,0,-1])

    G_x = sep_convolution(image, kernel_pt1, kernel_pt2)
    G_y = sep_convolution(image, kernel_pt2, kernel_pt1)

    G = np.hypot(G_x, G_y)
    G = G / G.max()
    theta = np.arctan2(G_y, G_x)

    return G, theta

def sep_convolution(image, kernel_pt1, kernel_pt2):
    ret_image = np.zeros(image.shape)
    pad_x, pad_y = (kernel_pt1.shape[0] // 2, kernel_pt2.shape[0] // 2)
    padded_image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')

    for (x, y), val in np.ndenumerate(image):
        ret_image[x,y] = np.dot(np.dot(padded_image[x:x+kernel_pt1.shape[0], y:y+kernel_pt2.shape[0]], kernel_pt1), kernel_pt2)

    return ret_image