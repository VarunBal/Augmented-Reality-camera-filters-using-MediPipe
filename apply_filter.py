import mediapipe as mp
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
import csv

SKIP_FRAMES = 2

VISUALIZE_FACE_POINTS = False

filters = {
    'anonymous':
        {'path': "filters/anonymous.png",
         'anno_path': "filters/anonymous_annotations.csv",
         'morph': True, 'animated': False, 'has_alpha': True},
    'anime':
        {'path': "filters/anime.png",
         'anno_path': "filters/anime_annotations.csv",
         'morph': True, 'animated': False, 'has_alpha': True},
    'jason-joker':
        {'path': "filters/jason-joker.png",
         'anno_path': "filters/jason-joker_annotations.csv",
         'morph': True, 'animated': False, 'has_alpha': True},
}

filter_name = "anonymous"
filter = filters[filter_name]


# detect facial landmarks in image
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0

def load_filter(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha


img1, img1_alpha = load_filter(filter['path'], filter['has_alpha'])

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


points1 = load_landmarks(filter['anno_path'])

if filter['morph']:
    # Find convex hull for delaunay triangulation using the landmark points
    hull1 = []
    hullIndex = cv2.convexHull(np.array(list(points1.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
        ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull1.append(points1[str(hullIndex[i][0])])

    # Find Delaunay triangulation for convex hull points
    sizeImg1 = img1.shape
    rect = (0, 0, sizeImg1[1], sizeImg1[0])
    dt = fbc.calculateDelaunayTriangles(rect, hull1)

    if len(dt) == 0:
        quit()

print("processed input image")

# process input from webcam or video file
cap = cv2.VideoCapture(0)

if filter['animated']:
    filter_cap = cv2.VideoCapture(filter['path'])

# Some variables
count = 0
isFirstFrame = True
sigma = 50

# The main loop
while True:

    ret, img2 = cap.read()
    if not ret:
        break
    else:

        # find landmarks after skipping SKIP_FRAMES number of frames
        if count % SKIP_FRAMES == 0:
            points2 = getLandmarks(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        # convert to float data type
        img1Warped = np.copy(img2)

        # if face is partially detected
        if not points2 or (len(points2) != 75):
            continue

        if VISUALIZE_FACE_POINTS:
            for idx, point in enumerate(points2):
                cv2.circle(img2, point, 2, (255, 0, 0), -1)
                cv2.putText(img2, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            cv2.imshow("landmarks", img2)

        # Find convex hull
        hull2 = []
        for i in range(0, len(hullIndex)):
            hull2.append(points2[hullIndex[i][0]])

        ################ Optical Flow and Stabilization Code #####################
        img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if isFirstFrame:
            hull2Prev = np.array(hull2, np.float32)
            img2GrayPrev = np.copy(img2Gray)
            isFirstFrame = False

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

        mask1 = np.zeros((img1Warped.shape[0], img1Warped.shape[1]), dtype=np.float32)
        mask1 = cv2.merge((mask1, mask1, mask1))
        img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

        # Warp the triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []

            for j in range(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])

            fbc.warpTriangle(img1, img1Warped, t1, t2)
            fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

        output = np.uint8(img1Warped)

        # Blur the mask before blending
        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

        mask2 = (255.0, 255.0, 255.0) - mask1

        # Perform alpha blending of the two images
        temp1 = np.multiply(output, (mask1 * (1.0 / 255)))
        temp2 = np.multiply(img2, (mask2 * (1.0 / 255)))
        result = temp1 + temp2

        result = np.uint8(result)

        cv2.imshow("Face Filter", result)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        count += 1

cap.release()
cv2.destroyAllWindows()
