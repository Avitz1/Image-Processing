from operations import *

if __name__ == "__main__":
    print(" ----------------------------- Question 1 -----------------------------")
    im1 = cv2.imread(r'Images\balls1.tif')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1_clean = clean_image_1(im1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im1_clean, cmap='gray', vmin=0, vmax=255)

    print(" ----------------------------- Question 2 -----------------------------")
    im2 = cv2.imread(r'Images\coins1.tif')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_cleaned = clean_image_canny(im2, 80, 255)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im2_cleaned, cmap='gray', vmin=0, vmax=255)

    print(" ----------------------------- Question 3 -----------------------------")
    im3 = cv2.imread(r'Images\balls1.tif')
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    im3_clean = clean_image_canny(im3, 90, 150)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im3_clean, cmap='gray', vmin=0, vmax=255)

    print(" ----------------------------- Question 4 -----------------------------")
    im4 = cv2.imread(r'Images\coins3.tif')
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    im4_clean = clean_hough(im4.copy())

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im4_clean, cmap='gray', vmin=0, vmax=255)


    print(" ----------------------------- Question 5 -----------------------------")
    im5_1 = cv2.imread(r'Images\boxOfChocolates1.tif')
    im5_1 = cv2.cvtColor(im5_1, cv2.COLOR_BGR2GRAY)
    edges = clean_image_canny(im5_1, 160, 200)
    im5_clean_1 = clean_hough_line(im5_1.copy(), edges, 180)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean_1, cmap='gray', vmin=0, vmax=255)



    im5_2 = cv2.imread(r'Images\boxOfchocolates2.tif')
    im5_2 = cv2.cvtColor(im5_2, cv2.COLOR_BGR2GRAY)
    edges = clean_image_canny(im5_2, 250, 250)
    im5_clean_2 = clean_hough_line(im5_2.copy(), edges, 90)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean_2, cmap='gray', vmin=0, vmax=255)



    im5_3 = cv2.imread(r'Images\boxOfchocolates2rot.tif')
    im5_3 = cv2.cvtColor(im5_3, cv2.COLOR_BGR2GRAY)
    edges = clean_image_canny(im5_3, 250, 280)
    im5_clean_3 = clean_hough_line(im5_3.copy(), edges, 100)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean_3, cmap='gray', vmin=0, vmax=255)



    x = 0


