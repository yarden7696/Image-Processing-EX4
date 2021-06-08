# ps2
import os
import numpy as np
from ex4_utils import *
import cv2


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():
    ## 1-a
    # Read images
    print("My ID - 207205972\n")
    i = 0
    L = cv2.imread(os.path.join('pair%d-L.png' % i), 0) / 255.0
    R = cv2.imread(os.path.join('pair%d-R.png' % i), 0) / 255.0
    # Display depth SSD
    displayDepthImage(L, R, (0, 5), method=disparitySSD)
    # Display depth NC
    displayDepthImage(L, R, (0, 5), method=disparityNC)

    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h = computeHomography(src, dst)
    print(h)
    #h_src = Homogeneous(src)
    #pred = h.dot(h_src.T).T

    #pred = unHomogeneous(pred)
    #print(np.sqrt(np.square(pred-dst).mean()))

    dst = cv2.imread(os.path.join('billBoard.jpg'), 0) / 255.0
    src = cv2.imread(os.path.join('car.jpg'), 0) / 255.0

    warpImag(src, dst)


if __name__ == '__main__':
    main()



