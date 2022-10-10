import traceback

import cv2
import numpy
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from pyntcloud import PyntCloud

def GSR():
    video = cv2.VideoCapture("cards.mp4")
    image = cv2.imread("transparent.png")

    output = cv2.VideoWriter("output.avi", cv2.VideoWriter.fourcc(*"MJPG"), 20.0, (640, 480))

    for fr in range(100):  # remove green from frames of video/image

        ret, frame = video.read()

        frame = cv2.resize(frame, (640, 480))
        image = cv2.resize(image, (640, 480))
        #
        # u_green = np.array([130, 255, 150])
        # l_green = np.array([0, 15, 0])
        #
        # mask = cv2.inRange(frame, l_green, u_green)
        # res = cv2.bitwise_and(frame, frame, mask=mask)
        #
        # f = frame - res
        # f = np.where(f == 0, image, f)

        # we just use an empty image for the purpose of this MCVE
        img = frame
        width = len(img[0])
        height = len(img)

        empty_img = numpy.zeros((height, width, 3), numpy.uint8)

        i = 0
        r = 0
        c = 0

        for line in img:
            c = 0

            for pixel in line:
                blue = pixel[0]
                green = pixel[1]
                red = pixel[2]

                if green != max(red, green, blue) or green < 35:
                    # this has a greenishy hue
                    empty_img.itemset((r, c, 0), 255)

                c += 1
            r += 1

        # outline/Canny of image
        # edges = cv2.Canny(frame, 100, 200)
        #
        # plt.subplot(121), plt.imshow(frame, cmap='gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(edges, cmap='gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        #
        # plt.show()

        # output.write(f)

        # cv2.imshow("video", frame)
        # cv2.imshow("mask", f)

    video.release()
    output.release()
    cv2.destroyAllWindows()


def split_video():
    video = cv2.VideoCapture("output.avi")

    imagesList = []

    success, image = video.read()
    count = 0
    try:
        while success:
            success, image2 = video.read()
            cv2.imwrite(r"imgout\frame%d.png" % count, image2)  # save frame as JPEG file
            # print('Read a new frame: ', success)
            imagesList.append(image)
            count += 1
    except:
        print("same error")

    return imagesList

