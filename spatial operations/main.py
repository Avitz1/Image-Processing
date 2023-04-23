from hw3_functions import *
import os

if __name__ == "__main__":
    # feel free to load different image than lena
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 1 ----------------------------------------------------------
    #add salt and pepper noise - low
    lena_sp_low = addSPnoise(lena_gray, 0.05)  # TODO - add low noise

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_low, cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - low")
    plt.subplot(2, 3, 4)
    lena_sp_low_med = cleanImageMedian(lena_sp_low, 2)
    cv2.imwrite("Images\cleanedIimg\lena_sp_low_med.tif", lena_sp_low_med)
    plt.imshow(cleanImageMedian(lena_sp_low, 2), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_sp_low, 2, 5), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_sp_low, 2, 5, 5), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  TODO: add explanation\n")

    # 2 ----------------------------------------------------------
    # add salt and pepper noise - high
    lena_sp_high = addSPnoise(lena_gray, 0.3)  # TODO - add low noise

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_high, cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - high")
    plt.subplot(2, 3, 4)
    plt.imshow(cleanImageMedian(lena_sp_high, 3), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_sp_high, 4, 15), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_sp_high, 4, 15, 15), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  TODO: add explanation \n")

    # 3 ----------------------------------------------------------
    # add gaussian noise - low
    lena_gaussian = addGaussianNoise(lena_gray, 10)  # TODO - add low noise

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian, cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - low")
    plt.subplot(2, 3, 4)
    plt.imshow(cleanImageMedian(lena_gaussian, 3), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_gaussian, 3, 8), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_gaussian, 7, 15, 15), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  TODO: add explanation \n")

    # 4 ----------------------------------------------------------
    # add gaussian noise - high
    lena_gaussian = addGaussianNoise(lena_gray, 30)  # TODO - add high noise

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian, cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - high")
    plt.subplot(2, 3, 4)
    plt.imshow(cleanImageMedian(lena_gaussian, 3), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_gaussian, 3, 1), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_gaussian, 7, 35, 35), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  TODO: add explanation \n")

    plt.show()