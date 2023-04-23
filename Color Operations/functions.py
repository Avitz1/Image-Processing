import cv2
import numpy as np
import matplotlib.pyplot as plt

def print_IDs():
	print("316327246, 209042589\n")

# function 1
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

def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax+1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')

def subtractionImage(im1, im2):
    # return value
    return abs(im2 - im1)

# function 2
def minkowski2Dist(im1,im2):
    # getting 2 histograms
    im1Histogram, binEdges1 = np.histogram(im1, bins=256, range=(0, 255))
    im2Histogram, binEdges2 = np.histogram(im2, bins=256, range=(0, 255))

    # converting them to float + normalize
    floatIm1H = im1Histogram.astype(np.float) / im1.size
    floatIm2H = im2Histogram.astype(np.float) / im2.size

    # getting minkowski distance
    temp = np.power((floatIm2H - floatIm1H), 2)
    d = np.power(np.sum(temp), 0.5)

    # return value
    return d

# function 3
def meanSqrDist(im1, im2):
	# return value
    return (np.sum(np.power((im1 - im2), 2)).astype(float) / im1.size)

# function 4
def sliceMat(im):
    # build matrix of zeroes
    Slices = np.zeros((im.size, 256), int)
    temp = np.ravel(im)

    # append 1 where we need to - each entry k in column i which represents a color
    for i in range(256):
        Slices[:, i] = temp == i

    # return value
    return Slices

# function 5
def SLTmap(im1, im2):
    # settings
    slicesIm1 = sliceMat(im1)
    vectorOfIm2 = np.ravel(im2)
    TM = np.zeros(256)

    # getting new color for each of original ones
    for i in range(256):
        if np.sum(slicesIm1[:, i]) != 0:
            TM[i] = np.matmul(vectorOfIm2, slicesIm1[:, i]).astype(float) / np.sum(slicesIm1[:, i])

    # return value
    return mapImage(im1, TM), TM

# function 6
def mapImage(im,tm):
    # creating array
    TMim = np.zeros((im.shape[0], im.shape[1]), np.uint8)

    # we are tone mapping image according to tm
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            TMim[i][j] = tm[im[i][j]]

    # return value
    return TMim

# function 7
def sltNegative(im):
    # return value
	return mapImage(im, np.array(range(255, -1, -1)))

# function 8
def sltThreshold(im, thresh):
    # set color vector
    colorVector = np.zeros(256)

    # update vector according to thresh
    i = 0
    while i < len(colorVector):
        if i > thresh:
            colorVector[i] = 255
        i += 1

    # map image according to color vector
    nim = mapImage(im, colorVector)

    # return value
    return nim

'''
# present histogram to screen
    plt.figure()
    plt.bar(binEdges2[:-1], im2Histogram, width=0.5, color='#0504aa', alpha=0.7)
    plt.xlim(min(binEdges2), max(binEdges2))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.title('Normal Distribution Histogram', fontsize=15)
    plt.show()
'''




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def print_IDs():
#     # print("123456789")
#     print("123456789+987654321\n")
#
#
# def contrastEnhance(im, range):
#     newContrast = range[1] - range[0]
#     oldContrast = np.amax(im) - np.amin(im)
#     a = float(newContrast/oldContrast)
#     b = range[0] - a*np.amin(im)
#     nim = (a * im + b).astype(np.uint8)
#     return nim, a, b
#
#
# def showMapping(old_range, a, b):
#     imMin = np.min(old_range)
#     imMax = np.max(old_range)
#     x = np.arange(imMin, imMax + 1, dtype=np.float)
#     y = a * x + b
#     plt.figure()
#     plt.plot(x, y)
#     plt.xlim([0, 255])
#     plt.ylim([0, 255])
#     plt.title('contrast enhance mapping')
#
#
# def minkowski2Dist(im1, im2):
#
#     hist_im2 = np.histogram(im2, bins=256, range=(0, 255))[0].astype(np.float)
#     hist_im1 = np.histogram(im1, bins=256, range=(0, 255))[0].astype(np.float)
#     hist_im1_normalized = hist_im1/im1.size
#     hist_im2_normalized = hist_im2/im2.size
#     distanceArray = hist_im1_normalized - hist_im2_normalized
#     return np.power(np.sum(np.power(distanceArray, 2)), 0.5)
#
# def meanSqrDist(im1, im2):
#
#     return (np.sum((np.power((im1 - im2), 2)))/im1.size).astype(np.float)
#
# def sliceMat(im):
#     pixelNum = im.size
#     raveled_im = np.ravel(im)
#     SL = np.zeros(shape=(pixelNum, 256))
#     for i in range(256):
#         SL[:, i] = raveled_im == i
#     return SL
#
#
# def SLTmap(im1, im2):
#     SL_im1 = sliceMat(im1)
#     im2_vectorized = np.ravel(im2)
#     TM = np.zeros(256)
#     for i in range(256):
#         if np.sum(SL_im1[:, i]) != 0:
#             TM[i] = np.dot(im2_vectorized, SL_im1[:, i])/np.sum(SL_im1[:, i])
#     return mapImage(im1, TM), TM
#
#
# def mapImage(im, tm):
#     TMim = np.zeros((im.shape[0], im.shape[1]), np.uint8)
#     for i in range(im.shape[0]):
#         for j in range(im.shape[1]):
#             TMim[i][j] = tm[im[i][j]]
#     return TMim
#
#
# def sltNegative(im):
#     return mapImage(im, range(255, 0, -1))
#
#
#
# def sltThreshold(im, thresh):
#     tm = np.zeros(256, np.uint8)
#     tm[thresh + 1:] = 255
#     return mapImage(im, tm)
