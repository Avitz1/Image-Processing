import numpy as np
from hw1_functions import *

if __name__ == "__main__":
    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    # read darkImage.tif
    path_image = r'Images\darkimage.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)
    # enhance contrast to maximum range
    enhanced_img, a, b = contrastEnhance(darkimg, [0, 255])
    # display images (original and enhanced)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')
    # print a,b
    print("a = {}, b = {}\n".format(a, b))
    # display mapping
    showMapping([np.min(darkimg), np.max(darkimg)], a, b)

    print("b ------------------------------------\n")
    # perform maximum contrast enhancing on an already enhanced image (from previous section)
    enhanced2_img, a, b = contrastEnhance(enhanced_img, [0, 255])
    print("enhancing an already enhanced image\n")
    # print a,b
    print("a = {}, b = {}\n".format(a, b))
    # display the difference between both images by using an image subtraction
    print("display the difference between both images by using an image subtraction")
    subtraction_img = subtractionImage(enhanced_img, enhanced2_img)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced2_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast2')
    plt.subplot(1, 3, 3)
    plt.imshow(subtraction_img, cmap='gray', vmin=0, vmax=255)
    plt.title('subtraction')

    print("c ------------------------------------\n")
    # read barbara.tif
    path_image = r'Images\barbara.tif'
    barbara = cv2.imread(path_image)
    barbara_gray = cv2.cvtColor(barbara, cv2.COLOR_BGR2GRAY)
    # show that distance between image and itself is 0
    mdist = minkowski2Dist(barbara_gray, barbara_gray)
    print("Minkowski dist between image and itself\n")
    print("d = {}\n".format(mdist))
    # plot the distance between the image and a contrast enhanced version of the image
    steps = [np.min(barbara_gray), np.max(barbara_gray)]
    k = (np.max(steps) - np.min(steps)) / 20
    contrast = []
    dists = []
    for i in range(1, 21, 1):
        # add item to contrast list
        contrast.append(round(i * k))
        # enhance contrast
        enhanced3_img, a, b = contrastEnhance(barbara_gray, [np.min(steps), np.min(steps) + np.max(contrast)])
	    # distance between the image and a contrast enhanced version of the image
        mdist = minkowski2Dist(barbara_gray, enhanced3_img)
        # add item to distances list
        dists.append(mdist)
    # display graph
    plt.figure()
    plt.plot(contrast, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")

    print("d ------------------------------------\n")
    # read fruit.tif
    path_image = r'Images\fruit.tif'
    fruit = cv2.imread(path_image)
    fruit_gray = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    # calculate sliceMat() for image
    Slices = sliceMat(fruit_gray)
    # creating a color vector and calculate sliceMat(im) * [0:255]
    colorVector = np.array(range(0, 256))
    matrixMultipication = np.reshape(np.matmul(Slices, np.transpose(colorVector)), (fruit_gray.shape[0], fruit_gray.shape[1]))
    # computationally prove that sliceMat(im) * [0:255] == im
    d = meanSqrDist(matrixMultipication, fruit_gray)
    print("prove that sliceMat(im) * [0:255] == im by showing 'meanSquareDistance' between them is {}\n".format(d))

    print("e ------------------------------------\n")
    # enhance contrast to maximum range
    enhanced4_img, a, b = contrastEnhance(darkimg, [0, 255])
    # find the tone map TM that creates this contrast enhancement
    enhanced4_img_gray = cv2.cvtColor(enhanced4_img, cv2.COLOR_BGR2GRAY)
    TMim, TM = SLTmap(darkimg_gray, enhanced4_img_gray)
    # compare contrast enhanced image with TMim = sliceMat(im) * TM
    d = meanSqrDist(enhanced4_img_gray, TMim)
    print("sum of diff between image and slices*[0..255] = {}".format(d))
	# display image and its tone mapped version
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(TMim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")

    print("f ------------------------------------\n")
    # produce the negative image of im
    negative_im = sltNegative(darkimg_gray)
    # display negative image
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")

    print("g ------------------------------------\n")
    # read RealLena.tif
    lena = cv2.imread(r"Images\\RealLena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    # produce a binary image by defining TM that performs thresholding
    thresh = 120
    thresh_im = sltThreshold(lena_gray, thresh)
    # display image
    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")

    print("h ------------------------------------\n")
    # choose 2 image (im1 and im2)
    im1 = lena_gray
    im2 = darkimg_gray
    # test SLTmap()
    SLTim, TM = SLTmap(im1, im2)
	# then print
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    # show that 'meanSqrDist' between SLTmap(im1,im2) and im2 < 'meanSqrDist' between im1 and im2
    # mean sqr dist between im1 and im2
    d1 = meanSqrDist(im1, im2)
    print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    # mean sqr dist between mapped image and im2
    d2 = meanSqrDist(SLTim, im2)
    print("mean sqr dist between mapped image and im2 = {}\n".format(d2))

    print("i ------------------------------------\n")
    # calculate the other direction
    SLTim1, TM1 = SLTmap(im2, im1)
    # prove that SLTmap is not symmetric by showing SLTmap(im1,im2) != SLTmap(im2,im1)
    d = meanSqrDist(SLTim, SLTim1)
    print("prove that SLTmap is not symmetric by showing 'MeanSqrDist' between SLTim, SLTim1 is {}".format(d))

    plt.show()






















# from hw1_functions import *
#
# if __name__ == "__main__":
#     # feel free to add/remove/edit lines
#
#     path_image = r'Images\darkimage.tif'
#     darkimg = cv2.imread(path_image)
#     darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)
#     print("Start running script  ------------------------------------\n")
#     print_IDs()
#     print("a ------------------------------------\n")
#     enhanced_img, a, b = contrastEnhance(darkimg, [0, 255])
#     # display images
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(darkimg)
#     plt.title('original')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
#     plt.title('enhanced contrast')
#
#     # print a,b
#     print("a = {}, b = {}\n".format(a, b))
#
#     # display mapping
#     showMapping([np.min(darkimg_gray), np.max(darkimg_gray)], a, b)  # add parameters
#
#     print("b ------------------------------------\n")
#     enhanced2_img, a, b = contrastEnhance(darkimg, [0, 255])  # add parameters
#     # print a,b
#     print("enhancing an already enhanced image\n")
#     print("a = {}, b = {}\n".format(a, b))
#
#     print("This it the Minkowski Distance between the enhanced and enhanced^2: " + str(minkowski2Dist(enhanced_img, enhanced2_img)) + "\nit is zero...\n")
#
#     print("c ------------------------------------\n")
#     mdist = minkowski2Dist(darkimg_gray, darkimg_gray)
#     print("Minkowski dist between image and itself\n")
#     print("d = {}\n".format(mdist))
#
#
#     old_range = [np.amin(darkimg_gray), np.amax(darkimg_gray)]
#     dists = []
#     step = (np.amax(darkimg_gray) - np.amin(darkimg_gray)) // 20
#     contrast = np.array([i for i in range(old_range[0], old_range[1]+1, step)])
#     i = 0
#     for k in range(old_range[0], old_range[1], step):
#         loop_image, a, b = contrastEnhance(darkimg_gray, [old_range[0], contrast[i]])
#         dists.append(minkowski2Dist(darkimg_gray, loop_image))
#         i += 1
#
#     plt.figure()
#     plt.plot(contrast, dists)
#     plt.xlabel("contrast")
#     plt.ylabel("distance")
#     plt.title("Minkowski distance as function of contrast")
#
#     print("d ------------------------------------\n")
#
#     d =  meanSqrDist(darkimg_gray, np.reshape(np.matmul(sliceMat(darkimg_gray), np.transpose(np.arange(256))), (darkimg_gray.shape[0], darkimg_gray.shape[1])))
#     print("we show the equality: sliceMat(im) * [0:255] == im by showing that the mean square distance between the RHS and LHS is {}\n".format(d))
#
#     print("e ------------------------------------\n")
#     enhanced_img, a, b = contrastEnhance(darkimg, [0, 255])
#     TM = np.fromfunction(lambda i: a*i + b, (256,), dtype=int)
#     TMim = mapImage(cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY), TM)
#     d = meanSqrDist(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY), TMim)
#     print("sum of diff between image and slices*[0..255] = {}".format(d))
#     # then display
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(darkimg)
#     plt.title("original image")
#     plt.subplot(1, 2, 2)
#     plt.imshow(TMim, cmap='gray', vmin=0, vmax=255)
#     plt.title("tone mapped")
#
#     print("f ------------------------------------\n")
#     negative_im = sltNegative(darkimg_gray)
#     plt.figure()
#     plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
#     plt.title("negative image using SLT")
#
#     print("g ------------------------------------\n")
#     thresh = 120  # play with it to see changes
#     lena = cv2.imread(r"Images\lena.tif")
#     lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
#     thresh_im = sltThreshold(lena_gray, thresh)
#
#     plt.figure()
#     plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
#     plt.title("thresh image using SLT")
#
#     print("h ------------------------------------\n")
#     im1 = lena_gray
#     im2 = darkimg_gray
#     SLTim, TM =  SLTmap(im1, im2)
#
#     # then print
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(im1)
#     plt.title("original image")
#     plt.subplot(1, 3, 2)
#     plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
#     plt.title("tone mapped")
#     plt.subplot(1, 3, 3)
#     plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
#     plt.title("tone mapped")
#
#     d1 =  meanSqrDist(im1, im2)
#     d2 =  meanSqrDist(SLTim, im2)
#     print("mean sqr dist between im1 and im2 = {}\n".format(d1))
#     print("mean sqr dist between mapped image and im2 = {}\n".format(d2))
#
#     print("i ------------------------------------\n")
#     d = meanSqrDist(SLTmap(lena_gray, darkimg_gray)[0], SLTmap(darkimg_gray, lena_gray)[0])
#     print("we show by a counterexample that SLTmap isn't symmetric.\n")
#     print("following is the MSD between SLTmap(lena_gray, darkimg_gray) and SLTmap(darkimg_gray, lena_gray):{}".format(d))
#
#     plt.show()