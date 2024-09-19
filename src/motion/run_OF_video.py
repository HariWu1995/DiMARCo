import cv2
import numpy as np


ARGUMENTS = dict(
       farneback = [0.5, 3, 15, 3, 5, 1.2, 0],
            rlof = [],
    lucas_kanade = [],
)


FUNCTIONS = dict(
       farneback =         cv2.calcOpticalFlowFarneback,
    lucas_kanade = cv2.optflow.calcOpticalFlowSparseToDense,
            rlof = cv2.optflow.calcOpticalFlowDenseRLOF,
)


def optical_flow_sparse(video_path, color: int = -1):

    # read the video
    cap = cv2.VideoCapture(video_path)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100, 
                         qualityLevel = 0.3, 
                          minDistance = 7, 
                            blockSize = 7, )

    # ONLY Lucas-Kanade algorithm available
    funct = cv2.calcOpticalFlowPyrLK
    params = dict( winSize = (15, 15),
                   maxLevel = 2,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create some random colors
    if not isinstance(color, int):
        color = np.random.randint(0, 255, (100, 3))
    elif color < 0 or color > 255:
        color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Pipeline
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = funct(old_gray, frame_gray, p0, None, **params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        if k == ord("c"):
            mask = np.zeros_like(old_frame)
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


def optical_flow_dense(video_path, method: str = 'lucas_kanade', to_gray: bool = False):

    # read the video
    cap = cv2.VideoCapture(video_path)

    # Query function & params
    funct = FUNCTIONS[method]
    params = ARGUMENTS[method]

    # Read the first frame
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = funct(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame


def optical_flow(video_path, method: str = 'lucas_kanade', is_dense: bool = True)

    if is_dense:
        method = method.lower()
        assert method in list(FUNCTIONS.keys()), \
            f"method = {method} is not supported!"
        optical_flow_dense(video_path, method, to_gray=(method != 'rlof'))
        
    else:
        optical_flow_sparse(video_path)

