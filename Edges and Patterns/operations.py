import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
import scipy



def clean_image_1(im):
    mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mask_y = np.rot90(mask_x)

    G_x = scipy.signal.convolve2d(im, mask_x * 0.125, mode="same")
    G_y = scipy.signal.convolve2d(im, mask_y * 0.125, mode="same")

    newX = np.power(G_x, 2)
    newY = np.power(G_y, 2)
    magnitude = np.sqrt(newX + newY).astype(np.uint8)

    theta_image = (np.arctan2(G_y, G_x) * 180 / np.pi).astype(np.uint8)
    ret, bw_img = cv2.threshold(magnitude , 14, 255, cv2.THRESH_BINARY)

    return bw_img

def clean_image_canny(im, min, max):
    # step 1 --------- clean image with gaussian ---------
    im_blurred = cv2.GaussianBlur(im, (7, 7), 0).astype(np.uint8)
    clean = cv2.Canny(im, min, max)
    return clean


def clean_hough(im):

    newIm = cv2.medianBlur(im, 3)
    circels = cv2.HoughCircles(newIm, cv2.HOUGH_GRADIENT, 1, 45, param1=425, param2=52, minRadius=0, maxRadius=0)
    circels = np.uint16(np.around(circels))
    for i in circels[0, :]:
        newIm = cv2.circle(im, (i[0], i[1]), i[2], (0, 255, 0), 2)
    return newIm


def clean_hough_line(im, edges, votes):

    lines = cv2.HoughLines(edges, 1, np.pi / 180, votes)
    for line in lines:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)

        x1 = int(a * rho + 1000 * (-b))
        y1 = int(b * rho + 1000 * (a))
        x2 = int(a * rho - 1000 * (-b))
        y2 = int(b * rho - 1000 * (a))

        cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return im
