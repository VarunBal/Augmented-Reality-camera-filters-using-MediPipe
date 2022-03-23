import cv2, time, math
import numpy as np
import faceBlendCommon as fbc
import colorCorrection as cc
import csv

SKIP_FRAMES = 2
FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 480

VISUALIZE_FACE_POINTS = True

DATA_PATH = "filters/"
filter_name = "anonymous"

# Processing input file
filename1 = DATA_PATH + f"{filter_name}.png"
annotation_file = DATA_PATH + f"{filter_name}_annotations.csv"

# Read the image and resize it
img1 = cv2.imread(filename1)
height, width = img1.shape[:2]
IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
img1 = cv2.resize(img1, None,
                  fx=1.0 / IMAGE_RESIZE,
                  fy=1.0 / IMAGE_RESIZE,
                  interpolation=cv2.INTER_LINEAR)
resized_height, resized_width = img1.shape[:2]

# load landmark points
with open(annotation_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    points1 = [None] * 68
    for i, row in enumerate(csv_reader):
        # skip head or empty line if it's there
        try:
            x, y = int(row[1]), int(row[2])
            x, y = (x / width) * resized_width, (y / height) * resized_height
            points1[int(row[0])] = (int(x), int(y))
        except ValueError:
            continue

img1 = np.float32(img1)

# Find convex hull for delaunay triangulation using the landmark points
hull1 = []
hullIndex = cv2.convexHull(np.array(points1), clockwise=False, returnPoints=False)
# addPoints = [[48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59]]  # Outer lips
# hullIndex = np.concatenate((hullIndex, addPoints))
addPoints = [[60], [61], [62], [63], [64], [65], [66], [67]]  # Inner lips
hullIndex = np.concatenate((hullIndex, addPoints))
# addPoints = [[27], [28], [29], [30], [31], [32], [33], [34], [35]]  # Nose
# hullIndex = np.concatenate((hullIndex, addPoints))
addPoints = [[36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47]]  # Eyes
hullIndex = np.concatenate((hullIndex, addPoints))
for i in range(0, len(hullIndex)):
    hull1.append(points1[hullIndex[i][0]])

# Find Delaunay triangulation for convex hull points
sizeImg1 = img1.shape
rect = (0, 0, sizeImg1[1], sizeImg1[0])
dt = fbc.calculateDelaunayTriangles(rect, hull1)

if len(dt) == 0:
    quit()

print("processed input image")

# process input from webcam or video file
# cap = cv2.VideoCapture(DATA_PATH + "sample-video.mp4")
cap = cv2.VideoCapture(0)

# Some variables for tracking time
count = 0
fps = 30.0
tt = time.time()
isFirstFrame = False
sigma = 50

# The main loop
while True:

    ret, img2 = cap.read()
    if not ret:
        break
    else:

        # Read each frame
        height, width = img2.shape[:2]
        IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
        img2 = cv2.resize(img2, None,
                          fx=1.0 / IMAGE_RESIZE,
                          fy=1.0 / IMAGE_RESIZE,
                          interpolation=cv2.INTER_LINEAR)

        # find landmarks after skipping SKIP_FRAMES number of frames
        if count % SKIP_FRAMES == 0:
            points2 = fbc.getLandmarks(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), FACE_DOWNSAMPLE_RATIO)

        # convert  float data type
        img1Warped = np.copy(img2)
        img1Warped = np.float32(img1Warped)

        # if face is partially detected
        if not points2 or (len(points2) != 68):
            continue

        if VISUALIZE_FACE_POINTS:
            for point in points2:
                cv2.circle(img2, point, 2, (255, 0, 0), -1)
            cv2.imshow("landmarks", img2)

        # Find convex hull
        hull2 = []
        for i in range(0, len(hullIndex)):
            hull2.append(points2[hullIndex[i][0]])

        ################ Optical Flow and Stabilization Code #####################
        img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if not isFirstFrame:
            isFirstFrame = True
            hull2Prev = np.array(hull2, np.float32)
            img2GrayPrev = np.copy(img2Gray)

        lk_params = dict(winSize=(101, 101), maxLevel=15,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        hull2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, hull2Prev, np.array(hull2, np.float32),
                                                      **lk_params)

        # Final landmark points are a weighted average of detected landmarks and tracked landmarks

        for k in range(0, len(hull2)):
            d = cv2.norm(np.array(hull2[k]) - hull2Next[k])
            alpha = math.exp(-d * d / sigma)
            hull2[k] = (1 - alpha) * np.array(hull2[k]) + alpha * hull2Next[k]
            hull2[k] = fbc.constrainPoint(hull2[k], img2.shape[1], img2.shape[0])

        # Update variables for next pass
        hull2Prev = np.array(hull2, np.float32)
        img2GrayPrev = img2Gray
        ################ End of Optical Flow and Stabilization Code ###############

        # Warp the triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []

            for j in range(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])

            fbc.warpTriangle(img1, img1Warped, t1, t2)

        ##################  Blending  #############################################
        img1Warped = np.uint8(img1Warped)
        # cv2.imshow("img1Warped", img1Warped)

        # Color Correction of the warped image so that the source color matches that of the destination
        # output = cc.correctColours(img2, img1Warped, points2)
        output = img1Warped

        # cv2.imshow("After color correction", output)

        # Create a Mask around the face
        re = cv2.boundingRect(np.array(hull2, np.float32))
        centerx = (re[0] + (re[0] + re[2])) / 2
        centery = (re[1] + (re[1] + re[3])) / 2

        hull3 = []
        for i in range(0, len(hull2) - len(addPoints)):
            # Take the points just inside of the convex hull
            hull3.append((0.95 * (hull2[i][0] - centerx) + centerx, 0.95 * (hull2[i][1] - centery) + centery))

        mask1 = np.zeros((img2.shape[0], img2.shape[1], 3), dtype=np.float32)
        hull3Arr = np.array(hull3, np.int32)

        cv2.fillConvexPoly(mask1, hull3Arr, (255.0, 255.0, 255.0), 16, 0)

        # Blur the mask before blending
        mask1 = cv2.GaussianBlur(mask1, (51, 51), 10)

        mask2 = (255.0, 255.0, 255.0) - mask1

        # cv2.imshow("mask1", np.uint8(mask1))
        # cv2.imshow("mask2", np.uint8(mask2))

        # Perform alpha blending of the two images
        temp1 = np.multiply(output, (mask1 * (1.0 / 255)))
        temp2 = np.multiply(img2, (mask2 * (1.0 / 255)))
        result = temp1 + temp2

        # cv2.imshow("temp1", np.uint8(temp1))
        # cv2.imshow("temp2", np.uint8(temp2))

        result = np.uint8(result)

        cv2.imshow("After Blending", result)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        count += 1

cap.release()
cv2.destroyAllWindows()
