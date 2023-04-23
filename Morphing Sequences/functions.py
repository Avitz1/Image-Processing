import numpy as np
import cv2
import matplotlib.pyplot as plt

def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()

def createMorphSequence(im1, im1_pts, im2, im2_pts, t_list, transformType):
    if transformType:
        im1_to_im2 = findProjectiveTransform(im1_pts, im2_pts)
        im2_to_im1 = findProjectiveTransform(im2_pts, im1_pts)
    else:
        im1_to_im2 = findAffineTransform(im1_pts, im2_pts)
        im2_to_im1 = findAffineTransform(im2_pts, im1_pts)

    size = [im1.shape[0], im1.shape[1]]
    ims = []
    unit_mat = np.eye(3)

    for t in t_list:
        # T12_t = (1 - t) * unit_mat + t * im1_to_im2
        # T21_t = t * unit_mat + (1 - t) * im2_to_im1
        # newIm1 = mapImage(im1, T12_t, (im2.shape[0], im2.shape[1]))
        # newIm2 = mapImage(im2, T21_t, (im1.shape[0], im1.shape[1]))
        nim = ((1 - t)*im1 + t * im2).astype(np.uint8)
        ims.append(nim)
    return ims
#
def mapImage(im, T, sizeOutIm):
    im_new = np.zeros((sizeOutIm[0], sizeOutIm[1]))

    # create meshgrid of all coordinates in new image [x,y]

    array_1 = np.arange(sizeOutIm[0])
    array_2 = [i for i in range(sizeOutIm[1])]

    x, y = np.meshgrid(array_2, array_1)
    # add homogenous coord [x,y,1]
    z = np.ones(sizeOutIm[0]*sizeOutIm[1], dtype=np.uint8)
    x = x.ravel()
    y = y.ravel()
    xy = np.vstack([x, y, z])




    # calculate source coordinates that correspond to [x,y,1] in new image
    Tpinv = np.linalg.pinv(T)
    xyz_source = np.matmul(Tpinv, xy)
    # find coordinates outside range and delete (in source and target)

    xyz_source[0] = xyz_source[0] / xyz_source[2]
    xyz_source[1] = xyz_source[1] / xyz_source[2]

    xx = xyz_source[0]
    yy = xyz_source[1]
    xMask = np.logical_or(xx < 0, xx >= sizeOutIm[0] - 1)
    yMask = np.logical_or(yy >= sizeOutIm[1] - 1, yy < 0)
    outOfRange = np.logical_not(np.logical_or(xMask, yMask))
    xx = xx[outOfRange]
    yy = yy[outOfRange]

    yy_, xx_ = np.meshgrid(np.arange(0, sizeOutIm[0]), np.arange(0, sizeOutIm[1]))
    xx_ = xx_.ravel()
    yy_ = yy_.ravel()
    xx_ = xx_[outOfRange]
    yy_ = yy_[outOfRange]

    floor_x_floor_y = np.floor([xx, yy]).astype(np.uint8)
    floor_x_ceil_y = np.array([np.floor(xx).astype(np.uint8), np.ceil(yy).astype(np.uint8)])
    ceil_x_floor_y = np.array([np.ceil(xx).astype(np.uint8), np.floor(yy).astype(np.uint8)])
    ceil_x_ceil_y = np.ceil([xx, yy]).astype(np.uint8)

    delta_X = np.reshape(xyz_source[0] - np.floor(xyz_source[0]), (sizeOutIm[0], sizeOutIm[1]))
    delta_y = np.reshape(xyz_source[1] - np.floor(xyz_source[1]), (sizeOutIm[0], sizeOutIm[1]))

    I_floor_x_floor_y = np.zeros((sizeOutIm[0], sizeOutIm[1]))
    I_floor_x_ceil_y = np.zeros((sizeOutIm[0], sizeOutIm[1]))
    I_ceil_x_floor_y = np.zeros((sizeOutIm[0], sizeOutIm[1]))
    I_ceil_x_ceil_y = np.zeros((sizeOutIm[0], sizeOutIm[1]))

    I_floor_x_floor_y[xx_, yy_] = im[floor_x_floor_y[0].astype(int), floor_x_floor_y[1].astype(int)]
    I_floor_x_ceil_y[xx_, yy_] = im[floor_x_ceil_y[0].astype(int), floor_x_ceil_y[1].astype(int)]
    I_ceil_x_floor_y[xx_, yy_] = im[ceil_x_floor_y[0].astype(int), ceil_x_floor_y[1].astype(int)]
    I_ceil_x_ceil_y[xx_, yy_] = im[ceil_x_ceil_y[0].astype(int), ceil_x_ceil_y[1].astype(int)]

    v1 = delta_y*I_floor_x_floor_y + (1 - delta_y)*I_floor_x_ceil_y
    v2 = delta_y*I_ceil_x_floor_y + (1 - delta_y)*I_ceil_x_ceil_y

    Nim = (delta_X*v1 + (1 - delta_X)*v2).astype(np.uint8)

    return np.transpose(Nim)

    # p = 0
    # for j in range(sizeOutIm[0]):
    #     for i in range(sizeOutIm[1]):
    #         if xyz_source[0][p] >= 0 and xyz_source[0][p] < (sizeOutIm[0]-1) and xyz_source[1][p] >= 0 and xyz_source[1][p] < (sizeOutIm[1]-1):
    #             floor_x= np.int(np.floor(xyz_source[0][p]))
    #             floor_y = np.int(np.floor(xyz_source[1][p]))
    #             ceil_x = np.int(np.ceil(xyz_source[0][p]))
    #             ceil_y = np.int(np.ceil(xyz_source[1][p]))
    #
    #             delta_x = xyz_source[0][p] - floor_x
    #             delta_y = xyz_source[1][p] - floor_y
    #             v1 = delta_y*im[floor_x][floor_y] + (1 - delta_y)*im[floor_x][ceil_y]
    #             v2 = delta_y*im[ceil_x][floor_y] + (1 - delta_y)*im[ceil_x][ceil_y]
    #             im_new[i][j] = (delta_x*v1 + (1 - delta_x)*v2).astype(np.uint8)
    #             im_new[i][j] = im[np.round(xyz_source[0][p]).astype(np.uint8)][np.round(xyz_source[1][p]).astype(np.uint8)]
    #         else:
    #             im_new[i][j] = 0
    #         p += 1
    #
    #
    # return np.uint8(im_new)

#     # create meshgrid of all coordinates in new image [x,y]
#
#
#     # add homogenous coord [x,y,1]
#
#
#     # calculate source coordinates that correspond to [x,y,1] in new image
#
#
#     # find coordinates outside range and delete (in source and target)
#
#
#     # interpolate - bilinear
#
#
#     # apply corresponding coordinates
#     # new_im [ target coordinates ] = old_im [ source coordinates ]


def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    X = np.array([[]], dtype=float)
    for i in range(0, N):
        array_row_i = np.array([[pointsSet1[i, 0], pointsSet1[i, 1], 0, 0, 1, 0, -(pointsSet1[i, 0] * pointsSet2[i, 0])
                           , -(pointsSet1[i, 1] * pointsSet2[i, 0])]])
        array_row_i_plus_1 = [[0, 0, pointsSet1[i, 0], pointsSet1[i, 1], 0, 1, -(pointsSet1[i, 0] * pointsSet2[i, 1])
                                  , -(pointsSet1[i, 1] * pointsSet2[i, 1])]]
        X = np.append(X, array_row_i)
        X = np.append(X, array_row_i_plus_1)

    x_tag = np.ravel(pointsSet2[:, [0, 1]])
    X = np.reshape(X, (2 * N, 8))
    X_pinv = np.linalg.pinv(X)
    T = np.matmul(X_pinv, x_tag)

    project = np.zeros((3, 3))
    project[0, 0] = T[0]
    project[0, 1] = T[1]
    project[0, 2] = T[4]
    project[1, 0] = T[2]
    project[1, 1] = T[3]
    project[1, 2] = T[5]
    project[2, 0] = T[6]
    project[2, 1] = T[7]
    project[2, 2] = 1


    return project


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    X = [[]]
    for i in range(0, N):
        array_row_i = [[pointsSet1[i, 0], pointsSet1[i, 1], 0, 0, 1, 0]]
        array_row_i_plus_1 = [[0, 0, pointsSet1[i, 0], pointsSet1[i, 1], 0, 1]]
        X = np.append(X, array_row_i)
        X = np.append(X, array_row_i_plus_1)

    x_tag = np.ravel(pointsSet2[:, [0, 1]])
    X = np.reshape(X, (2 * N, 6))
    T = np.matmul(np.linalg.pinv(X), x_tag)

    affine = np.zeros((3, 3))
    affine[0, 0] = T[0]
    affine[0, 1] = T[1]
    affine[0, 2] = T[4]
    affine[1, 0] = T[2]
    affine[1, 1] = T[3]
    affine[1, 2] = T[5]
    affine[2, 2] = 1

    return affine


def getImagePts(im1, im2,varName1,varName2, nPoints):
    plt.imshow(im1, cmap='gray')
    imagePts1 = np.round(np.array(plt.ginput(nPoints, timeout=60))).astype(np.float)
    plt.imshow(im2, cmap='gray')
    imagePts2 = np.round(np.array(plt.ginput(nPoints, timeout=60))).astype(np.float)
    imagePts1 = np.append(np.fliplr(imagePts1), np.ones((nPoints, 1), dtype=np.float), axis=1)
    imagePts2 = np.append(np.fliplr(imagePts2), np.ones((nPoints, 1), dtype=np.float), axis=1)
    np.save(varName1+".npy", imagePts1)
    np.save(varName2+".npy", imagePts2)


