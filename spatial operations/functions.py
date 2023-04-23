import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def addSPnoise(im, p):
    sp_noise_im = im.copy()
    no_pixels_to_noise = int(im.shape[0]*im.shape[1]*p)
    points_for_noise = random.sample(range(im.shape[0]*im.shape[1]), no_pixels_to_noise)
    sp_noise_im = np.ravel(sp_noise_im)
    sp_noise_im[points_for_noise[0: no_pixels_to_noise//2]] = 0
    sp_noise_im[points_for_noise[no_pixels_to_noise//2:]] = 255
    return np.reshape(sp_noise_im, (im.shape[0], im.shape[1]))


def addGaussianNoise(im, s):
    gaussian_noise_im = im.copy()
    noise_matrix = np.random.normal(0, s, (im.shape[0], im.shape[1]))
    gaussian_noise_im = (noise_matrix + gaussian_noise_im.astype(np.float)).astype(np.uint8)
    return gaussian_noise_im


def cleanImageMedian(im, radius):
    median_im = im.copy()
    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, im.shape[1] - radius):
            median_im[i][j] = np.median(im[i - radius:i + radius + 1, j - radius: j + radius + 1])
    return median_im


def cleanImageMean(im, radius, maskSTD):
    cleaned_im = im.copy().astype(np.float)
    gaussian_mask = np.fromfunction(lambda i, j: np.exp(-(np.divide(pow(i - radius, 2) + pow(j - radius, 2),
                                                                    2*pow(maskSTD, 2)))), (2*radius + 1, 2*radius + 1), dtype=np.float)
    gaussian_mask = np.divide(gaussian_mask, np.sum(gaussian_mask))
    cleaned_im = convolve2d(cleaned_im, gaussian_mask, mode='same')
    cleaned_im = cleaned_im.astype(np.uint8)
    return cleaned_im


def bilateralFilt(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()

    gs = np.fromfunction(
        lambda x, y: np.exp(-(np.divide(pow(x - radius, 2) + pow(y - radius, 2), 2 * pow(stdSpatial, 2)))),
        (2 * radius + 1, 2 * radius + 1), dtype=np.float)
    gs = np.divide(gs, np.sum(gs))

    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, im.shape[1] - radius):
            window = im[i - radius: i + radius + 1, j - radius: j + radius + 1].astype(np.float)

            gi = np.exp(-np.divide(np.power(np.full((2 * radius + 1, 2 * radius + 1), im[i][j], dtype=float) - window, 2),
                        2 * pow(stdIntensity, 2)))
            gi = np.divide(gi, np.sum(gi))

            bilateral_im[i][j] = np.divide(np.sum(gi*gs*window), np.sum(gi*gs))

    return bilateral_im.astype(np.uint8)


