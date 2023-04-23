from hw2_functions import *

if __name__ == '__main__':
    path_image1 = r'Images\Face1.tif'
    face1 = cv2.imread(path_image1)
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    path_image2 = r'Images\Face2.tif'
    face2 = cv2.imread(path_image2)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    #getImagePts(face1_gray, face2_gray, "imagePts1", "imagePts2",12)
    imagePts1 = np.load('imagePts1.npy')
    imagePts2 = np.load('imagePts2.npy')
    y = np.linalg.matrix_rank(imagePts1)
    tran = findProjectiveTransform(imagePts1, imagePts2)
    face2_gray = mapImage(face1_gray, tran, [face1.shape[0], face1.shape[1]])
    plt.imshow(face2_gray, cmap='gray')

    t_list = np.linspace(0, 1, 20)

    im_list = createMorphSequence(face1_gray, imagePts1, face2_gray, imagePts2, t_list, False)

    writeMorphingVideo(im_list, 'check2')



