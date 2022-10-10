import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from pyntcloud import PyntCloud

# minimum and maximum HSV values to be removed for the green colour
MIN_HSV_GREEN = (100, 80, 70)
MAX_HSV_GREEN = (185, 255, 255)

imageList = [None] * 100  # empty list to contain GSR images ready for disparity processing

# video file to be read
vid = r"videos\amber-low\30fps.mp4"


def RGBtoHSV(R, G, B):
    max_value = max(R, G, B)  # maximum value of the three parameters
    min_value = min(R, G, B)
    V = max_value  # value is described as the distance between 0 and the highest value (255)
    if min_value == max_value:
        return 0.0, 0.0, V
    S = (V - min_value) / V  # saturation is described as the ratio of the minimum RGB value taken away from v and
    # divided by v

    R_value = (V - R) / (V - min_value)
    G_value = (V - G) / (V - min_value)
    B_value = (V - B) / (V - min_value)

    # hue is defined as the ratio between red green and blue compared to v read documentation under the RGB to HSV
    # section (5.1) in the final report document to find out how this is calculated
    if R == V:
        H = B_value - G_value
    elif G == V:
        H = 2.0 + R_value - B_value
    else:
        H = 4.0 + G_value - R_value
    # at the end the hue is divided by 6 as there are 6 sextants in a hexagon and is modded by 1 as H consists of 0
    # and 360 degrees only (0,1)
    H = (H / 6.0) % 1.0
    return H, S, V


def GSR():
    split_video()
    count = 1  # for some reason the function crashed the program sometimes when it started at 0 so I made it start at 1
    # this causes error messages in the console because the rest of the program operates normally but I have made sure
    # that the program keeps running regardless of this

    for fr in range(100):  # remove green from frames of video/image
        # Load image and convert it to RGBA, so it contains alpha channel
        im = Image.open(r"imgout\frame%d.png" % count)
        im = im.convert('RGBA')

        # Go through all pixels and turn each 'green' pixel to transparent
        pix = im.load()
        width, height = im.size
        for x in range(width):
            for y in range(height):
                R, G, B, A = pix[x, y]
                H_ratio, S_ratio, V_ratio = RGBtoHSV(R / 255.0, G / 255.0, B / 255.0)
                H, S, V = (H_ratio * 360, S_ratio * 255, V_ratio * 255)

                H_min, S_min, V_min = MIN_HSV_GREEN
                H_max, S_max, V_max = MAX_HSV_GREEN
                if H_min <= H <= H_max and S_min <= S <= S_max and V_min <= V <= V_max:
                    pix[x, y] = (0, 0, 0, 0)

        im.save(r"imgout\noGS\img" + str(count) + '.png')
        count += 1


def split_video():
    video = cv2.VideoCapture(vid)

    success, image = video.read()
    count = 1
    while success:
        cv2.imwrite(r"imgout\frame%d.png" % count, image)  # save frame as JPEG file
        success, image = video.read()
        print('Read a new frame: ', success)
        count += 1


def readImages(amount):
    for i in range(amount):
        imageList[i] = cv2.imread(r"imgout\noGS\img%d.png" % i)

    return imageList


def readImagesD(amount):
    try:
        for i in range(amount):
            imageList[i] = cv2.imread(r"depth_maps\img%d.png" % i)

    except:
        print("testo")
    return imageList


# DMC - Depth Map Creator
def DMC(imageSequence):
    i = 1
    for loc in imageSequence:
        try:
            # replace with loop with file locations
            left = imageSequence[i]
            right = imageSequence[i + 1]

            # resize frames to 640 by 480 pixels from 1920 by 1080
            cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.resize(right, (640, 480), interpolation=cv2.INTER_LINEAR)

            # set the disparity computation settings
            # (I found these to be the best for my setup, although there was not much of a difference)
            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

            # initialising left and right image to be compared (right is the one left is compared to)
            new_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            new_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            # calculate disparity map and save the image to a file
            disparity = stereo.compute(new_left, new_right)
            plt.axis("off")
            plt.imshow(disparity, "gray")
            plt.savefig(r"depth_maps\img%d.png" % i)
            print("saved %d" % i)
            plt.close()
            i = i + 1
        except:
            traceback.print_exc()


def reconstruct(amount):

    # loop through each disparity map set to only the first 100 frames for time efficiency
    for i in range(amount):
        try:
            # get the original colour image for later mapping of colours to corresponding pixels(GSR images)
            colourImg = Image.open(r"imgout\noGS\img%d.png" % i)
            colourPixels = colourImg.convert("RGB")

            # predefining the colour and indices array for cleaner look
            colourArray = np.array(colourPixels.getdata()).reshape((colourImg.height, colourImg.width) + (3,))
            indicesArray = np.moveaxis(np.indices((colourImg.height, colourImg.width)), 0, 2)
            imageArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))

            df = pd.DataFrame(imageArray, columns=["x", "y", "red", "green", "blue"])

            depthImg = Image.open(r"depth_maps\img%d.png" % i).convert('L')
            depthArray = np.array(depthImg.getdata())
            df.insert(loc=2, column='z', value=pd.Series(depthArray))

            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)
            df[['red', 'green', 'blue']] = df[['red', 'green', 'blue']].astype(np.uint)
            cloud = PyntCloud(df)
            cloud.to_file(r"clouds\cloud%d.ply" % i, also_save=["mesh", "points"], as_text=True)
        except:
            traceback.print_exc()


if __name__ == "__main__":
    GSR()
    imageList = readImages(99)  # for in range starts with 0 so has to be 99 to get 100 elements
    DMC(imageList)
    reconstruct(99)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.create_from_depth_image(o3d.io.read_image(r"depth_maps\img1.png"), PinholeCameraIntrinsic(
    #     PinholeCameraIntrinsicParameters.PrimeSenseDefault), np.identity(4))
    # cv2.reprojectImageTo3D(r"depth_maps\img1.png", cv2.stereoRectify())
