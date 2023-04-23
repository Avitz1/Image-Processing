# import numpy as np
# from scipy.signal import convolve2d
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

def print_IDs():
    print("209042589, 316327246\n")

# our function (1)
def cleanImageMedian(im, radius):
    # copy image
    median_im = im.copy()

    # filtering each pixel through a square mask of (radius * 2 + 1)^2
    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, im.shape[1] - radius):
            median_im[i][j] = np.median(im[i - radius : i + radius + 1, j - radius : j + radius + 1])

    # return value
    return median_im

# our function (2)
def contrastEnhance(im, range):
    # getting parameter a
    oldContrast = np.max(im) - np.min(im)
    newContrast = range[1] - range[0]
    a = newContrast / oldContrast

    # getting parameter b
    b = range[1] - a * np.max(im)

    # creating new enhanced image
    nim = (a * im + b).astype(np.uint8)

    # return values
    return nim, a, b

# function 1
def clean_im1(im):
    # extracting the normal picture out of the three
    baby1 = im[20: 130, 6: 111]

    # changing order of pixels inside each row / column
    pts1 = np.fliplr(np.float32([[0, 0], [0, 104], [109, 104], [109, 0]]))

    # apply perspective / projective transform algorithm
    pts_baby2 = np.load('pts_baby2.npy')
    pts_baby3 = np.load('pts_baby3.npy')
    matrix_baby2 = cv2.getPerspectiveTransform(pts_baby2, pts1)
    matrix_baby3 = cv2.getPerspectiveTransform(pts_baby3, pts1)
    baby2 = cv2.warpPerspective(im, matrix_baby2, (105, 110)).astype(np.uint8)
    baby3 = cv2.warpPerspective(im, matrix_baby3, (105, 110)).astype(np.uint8)

    # taking the median per pixel
    clean_im = np.array([baby1, baby2, baby3])
    clean_im = np.median(clean_im, axis=0).astype(np.uint8)

    # cleaning leftovers with the smallest radius possible.
    clean_im = cleanImageMedian(clean_im, 1)

    # lastly, rescaling the image (we cleaned it before scaling it because we didn't want to enlarge the noise too)
    pts2 = np.fliplr(np.float32([[0, 0], [0, 255], [255, 255], [255, 0]]))
    scaling_mat = cv2.getPerspectiveTransform(pts1, pts2)
    clean_im = cv2.warpPerspective(clean_im, scaling_mat, (255, 255)).astype(np.uint8)

    # return value
    return clean_im

# function 2
def clean_im2(im):
    # fourier transform - from image domain to frequency domain
    img_fourier = np.fft.fftshift(np.fft.fft2(im))

    # canceling abnormal intensities by replacing their values with 0
    img_fourier[124][100] = 0
    img_fourier[132][156] = 0

    # returning to image domain (fourier transform on 'img_fourier')
    clean_im = np.abs(np.fft.ifft2(img_fourier))

    # return value
    return clean_im.astype(np.uint8)

# function 3
def clean_im3(im):
    # creating mask: identity + derive
    # -1 -1 -1     0 0 0     -1 -1 -1
    # -1 +9 -1  =  0 1 0  +  -1 +8 -1
    # -1 -1 -1     0 0 0     -1 -1 -1
    differentiating_kernel = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])

    # applying the mask on image
    cleaned_im = convolve2d(im, differentiating_kernel, mode='same').astype(np.uint8)

    # enhancing it
    cleaned_im, _, _ = contrastEnhance(cleaned_im, [0, 255])

    # the following lines are to handle the image margins
    cleaned_im[0] = cleaned_im[1]
    cleaned_im[:, 0] = cleaned_im[:, 1]
    cleaned_im[255] = cleaned_im[254]
    cleaned_im[:, 255] = cleaned_im[:, 254]

    # return value
    return cleaned_im

# function 4
def clean_im4(im):
    shift_mat = np.zeros(im.shape, np.uint8)
    shift_mat[0][0] = 1
    shift_mat[4][79] = 1

    img_fourier = np.fft.fft2(im)
    shift_fourier = np.fft.fft2(shift_mat)
    shift_fourier[np.abs(shift_fourier) < 0.01] = 1
    result = img_fourier / shift_fourier

    clean_im = 2 * abs(np.fft.ifft2(result)).astype(np.uint8)

    return clean_im

# function 5
def clean_im5(im):
    # copy image
    clean_im = im.copy()

    # saving stars for later - we don't want to be accidentally considered as noise
    stars = im[0:91, 0:146]

    # median for every pixel
    for i in range(im.shape[0]):
        for j in range(7, im.shape[1] - 7):
            clean_im[i][j] = np.median(im[i, j - 7: j + 7 + 1])

    # assigning the stars to their proper place
    clean_im[0:91, 0:146] = stars

    # enhancing image
    clean_im, _, _ = contrastEnhance(clean_im, (0, 255))

    # return value
    return clean_im

# function 6
def clean_im6(im):
    # fourier transform
    img_fourier = np.fft.fftshift(np.fft.fft2(im))

    # creating H mask with DC = 1
    H = np.ones(im.shape, np.uint8)
    H[107: 148, 109: 148] = 2.5
    H[128][128] = 1

    # convolution in the image domain = multiplying in the frequency domain
    clean_im = abs(np.fft.ifft2(H * img_fourier)).astype(np.uint8)

    # return value
    return clean_im

# function 7
def clean_im7(im):
    shift_mat = np.zeros(im.shape, np.uint8)
    shift_mat[0][0:10] = 1

    img_fourier = np.fft.fft2(im)
    shift_fourier = np.fft.fft2(shift_mat)
    shift_fourier[np.abs(shift_fourier) < 0.01] = 1
    result = img_fourier / shift_fourier

    clean_im = 2 * abs(np.fft.ifft2(result)).astype(np.uint8)
    clean_im, _, _ =contrastEnhance(clean_im, (0, 255))

    return clean_im

# function 8
def clean_im8(im):
    clean_im, _, _ = contrastEnhance(im, (0, 255))
    return clean_im


















# import matplotlib.pyplot as plt
# import cv2
#
# def print_IDs():
#     print("209042589, 316327246 \n")
#
# def contrastEnhance(im, range):
#     # getting parameter a
#     oldContrast = np.max(im) - np.min(im)
#     newContrast = range[1] - range[0]
#     a = newContrast / oldContrast
#
#     # getting parameter b
#     b = range[1] - a * np.max(im)
#
#     # creating new enhanced image
#     nim = (a * im + b).astype(np.uint8)
#
#     # return values
#     return nim, a, b
#
# def cleanImageMedian(im, radius):
#     median_im = im.copy()
#     for i in range(radius, im.shape[0] - radius):
#         for j in range(radius, im.shape[1] - radius):
#             median_im[i][j] = np.median(im[i - radius:i + radius + 1, j - radius: j + radius + 1])
#     return median_im
#
# def clean_im1(im):
#     baby1 = im[20: 130, 6: 111]
#
#     # The following lines are used to get new points
#     # plt.imshow(im, cmap='gray')
#     # pts_baby2 = np.round(np.array(plt.ginput(4, timeout=60))).astype(np.float32)
#     # plt.imshow(im, cmap='gray')
#     # pts_baby3 = np.round(np.array(plt.ginput(4, timeout=60))).astype(np.float32)
#     # np.save("pts_baby2.npy", pts_baby2)
#     # np.save("pts_baby3.npy", pts_baby3)
#
#     pts_baby2 = np.load('pts_baby2.npy')
#     pts_baby3 = np.load('pts_baby3.npy')
#
#     pts1 = np.fliplr(np.float32([[0, 0], [0, 104], [109, 104], [109, 0]]))
#
#     # Apply Perspective Transform Algorithm
#     matrix_baby2 = cv2.getPerspectiveTransform(pts_baby2, pts1)
#     matrix_baby3 = cv2.getPerspectiveTransform(pts_baby3, pts1)
#
#     baby2 = cv2.warpPerspective(im, matrix_baby2, (105, 110)).astype(np.uint8)
#     baby3 = cv2.warpPerspective(im, matrix_baby3, (105, 110)).astype(np.uint8)
#
#     # taking the median per pixel
#     clean_im = np.array([baby1, baby2, baby3])
#     clean_im = np.median(clean_im, axis=0).astype(np.uint8)
#
#     # Cleaning leftovers with the smallest radius.
#     clean_im = cleanImageMedian(clean_im, 1)
#
#     # Lastly, rescaling the image. we cleaned it before because we didn't want to enlarge the noise too.
#     pts2 = np.fliplr(np.float32([[0, 0], [0, 255], [255, 255], [255, 0]]))
#     scaling_mat = cv2.getPerspectiveTransform(pts1, pts2)
#     clean_im = cv2.warpPerspective(clean_im, scaling_mat, (255, 255)).astype(np.uint8)
#
#     return clean_im
#
#
# def clean_im2(im):
#
#     img_fourier = np.fft.fftshift(np.fft.fft2(im))
#     img_fourier[124][100] = 0
#     img_fourier[132][156] = 0
#     clean_im = np.abs(np.fft.ifft2(img_fourier))
#     return clean_im.astype(np.uint8)
#
#
# def clean_im3(im):
#     differentiating_kernel = np.array([[-1, -1, -1],
#                                        [-1, 9, -1],
#                                        [-1, -1, -1]])
#
#     cleaned_im = convolve2d(im, differentiating_kernel, mode='same').astype(np.uint8)
#
#     cleaned_im, _, _ = contrastEnhance(cleaned_im, [0, 255])
#
#     # The following lines are to handle the image margins
#     cleaned_im[0] = cleaned_im[1]
#     cleaned_im[:, 0] = cleaned_im[:, 1]
#     cleaned_im[255] = cleaned_im[254]
#     cleaned_im[:, 255] = cleaned_im[:, 254]
#
#     return cleaned_im
#
#
# def clean_im4(im):
#     shift_mat = np.zeros(im.shape, np.uint8)
#     shift_mat[0][0] = 1
#     shift_mat[4][79] = 1
#
#     img_fourier = np.fft.fft2(im)
#     shift_fourier = np.fft.fft2(shift_mat)
#     shift_fourier[np.abs(shift_fourier) < 0.01] = 1
#     result = img_fourier / shift_fourier
#
#     clean_im = 2 * abs(np.fft.ifft2(result)).astype(np.uint8)
#     return clean_im
#
#
# def clean_im5(im):
#     clean_im = im.copy()
#     stars = im[0:91, 0:146]
#     for i in range(im.shape[0]):
#         for j in range(7, im.shape[1] - 7):
#             clean_im[i][j] = np.median(im[i, j - 7: j + 7 + 1])
#
#     clean_im[0:91, 0:146] = stars
#     clean_im, _, _ = contrastEnhance(clean_im, (0, 255))
#     return clean_im
#
# def clean_im6(im):
#     img_fourier = np.fft.fftshift(np.fft.fft2(im))
#     H = np.ones(im.shape, np.uint8)
#     H[107:148, 109: 148] = 2.5
#     H[128][128] = 1  # In order to keep the DC as it was
#     clean_im = abs(np.fft.ifft2(H * img_fourier)).astype(np.uint8)
#     return clean_im
#
#
# def clean_im7(im):
#     clean_im = 0
#     shift_mat = np.zeros(im.shape, np.uint8)
#     shift_mat[0][0:10] = 1
#
#     img_fourier = np.fft.fft2(im)
#     shift_fourier = np.fft.fft2(shift_mat)
#     shift_fourier[np.abs(shift_fourier) < 0.01] = 1
#     result = img_fourier / shift_fourier
#
#     clean_im = 2 * abs(np.fft.ifft2(result)).astype(np.uint8)
#     clean_im, _, _ =contrastEnhance(clean_im, (0, 255))
#     return clean_im
#
#
# def clean_im8(im):
#     clean_im, _, _ = contrastEnhance(im, (0, 255))
#     return clean_im