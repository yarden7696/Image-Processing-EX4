import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters


def disparity_help(img_l: np.ndarray,img_r: np.ndarray, disp_range: (int, int), k_size: int) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray):

    height, width = img_r.shape
    disp_map = np.zeros((height, width, disp_range[1]))

    mean_left = np.zeros((height, width))
    mean_right = np.zeros((height, width))

    # calc average of our window using uniform_filter
    filters.uniform_filter(img_l, k_size, mean_left)
    filters.uniform_filter(img_r, k_size, mean_right)

    norm_left = img_l - mean_left  # normalized left image
    norm_right = img_r - mean_right  # normalized right image

    return disp_map, mean_left, mean_right, height, width, norm_left, norm_right


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    disp_map, mean_left, mean_right, height, width, norm_left, norm_right = disparity_help(img_l,img_r, disp_range,k_size)

    for i in range(disp_range[1]):
        rImg_shift = np.roll(norm_right, i)  # moving i element to the front
        filters.uniform_filter(norm_left * rImg_shift, k_size, disp_map[:, :, i])
        disp_map[:, :, i] = disp_map[:, :, i] ** 2  # (Li-Ri)^2

    res = np.argmax(disp_map, axis=2)  # taking best depth
    return res
    pass


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    disp_map, mean_left, mean_right, height, width, norm_left, norm_right = disparity_help(img_l,img_r, disp_range,k_size)

    sigma_l = np.zeros((height, width))
    sigma_r = np.zeros((height, width))
    sigma = np.zeros((height, width))

    # Calculate the average of each pixel in a (k_size)^2 window
    filters.uniform_filter(norm_left * norm_left, k_size, sigma_l)

    for i in range(disp_range[1]):
        rImg_shift = np.roll(norm_right, i-disp_range[0])  # moving i element to the front
        filters.uniform_filter(norm_left * rImg_shift, k_size, sigma)  # calc sigma using uniform_filter
        filters.uniform_filter(rImg_shift * rImg_shift, k_size, sigma_r)  # calc sigma r using uniform_filter
        sqr = np.sqrt(sigma_r * sigma_l)
        disp_map[:, :, i] = sigma / sqr

    ans = np.argmax(disp_map, axis=2)  # taking best depth
    return ans


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """

    A = np.zeros((2 * len(src_pnt), 9))  # A is (2n x 9) mat
    for pos in range(len(src_pnt)):
        # In each iteration we fill 2 rows.
        # src_pnt [pos] [0] is Xi and dst_pnt [pos] [0] is 'Xi
        # Same for Yi and 'Yi (for example- opposite [pos][0] to [0][pos] )
        A[pos * 2:pos * 2 + 2] = np.array([[-src_pnt[pos][0], -src_pnt[pos][1], -1, 0, 0, 0,
                                       src_pnt[pos][0] * dst_pnt[pos][0], src_pnt[pos][1] * dst_pnt[pos][0], dst_pnt[pos][0]],
                                       [0, 0, 0, -src_pnt[pos][0], -src_pnt[pos][1], -1,
                                       src_pnt[pos][0] * dst_pnt[pos][1],src_pnt[pos][1] * dst_pnt[pos][1], dst_pnt[pos][1]]])

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(vh)  # vh^T

    H = V[:, -1].reshape(3, 3)
    H /= V[:, -1][-1]

    err = 0
    for pos in range(len(src_pnt)):
        Homogeneous = H.dot(np.array([src_pnt[pos, 0], src_pnt[pos, 1], 1]))
        Homogeneous /= Homogeneous[2]
        err += np.sqrt(sum(Homogeneous[0:-1] - dst_pnt[pos]) ** 2)

    return H, err
    pass



def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))  # print to the console the locations

        plt.plot(x, y, '*r')  # '*r' colored the chosen point
        dst_p.append([x, y])  # adding the chosen points to the dst_p array

        if len(dst_p) == 4:  # caz we need 4 points to compute homography
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    srcPoints = []
    secFig = plt.figure()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))  # print to the console the locations

        plt.plot(x, y, '*r')  # '*r' colored the chosen point
        srcPoints.append([x, y])  # adding the chosen points to the dst_p array

        if len(srcPoints) == 4:  # caz we need 4 points to compute homography
            plt.close()
        plt.show()

    # display image 2
    cid_sec = secFig.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    srcPoints = np.array(srcPoints)

    src0 = src_img.shape[0]
    src1 = src_img.shape[1]
    homography, err = computeHomography(srcPoints, dst_p)
    for Yi in range(src0):
        for Xi in range(src1):
            A_h = np.array([Xi, Yi, 1])
            A_h = homography.dot(A_h)  # inner product between homography matrix and [Xi, Yi, 1]
            A_h /= A_h[2]  # div the second row
            dst_img[int(A_h[1]), int(A_h[0])] = src_img[Yi, Xi]

    plt.imshow(dst_img)
    plt.show()
    pass








