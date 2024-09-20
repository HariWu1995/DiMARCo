import cv2
import numpy as np

from PIL import Image


PI = np.pi


def visualize_sparse_flow(frame: np.ndarray, points_1: np.ndarray, points_2: np.ndarray):
     
    # Create some random colors
    if not isinstance(color, int):
        color = np.random.randint(0, 255, (100, 3))
    elif color < 0 or color > 255:
        color = np.random.randint(0, 255, (100, 3))

    # Create a canvas image for drawing purposes
    frame = np.stack([frame] * 3, axis=0)
    canvas = np.zeros_like(frame.shape)

    radius = 5
    for i, (track_1, track_2) in enumerate(zip(points_1, points_2)):
        a, b = track_1.ravel()
        c, d = track_2.ravel()
        canvas = cv2.line(canvas, (a, b), (c, d), color[i].tolist(), thickness=2)
        frame = cv2.circle(frame, (a, b), radius, color[i].tolist(), thickness=-1)

    return cv2.add(frame, canvas)


def visualize_dense_arrow(flow: np.ndarray, image: np.ndarray = None,
                          threshold: float = 2., step: int = None):
    
    h, w = flow.shape[:2]

    if step is None:
        step = int(min(h, w) / 100)
    if step < 1:
        step = 1

    if image is None:
        canvas = np.zeros([h, w])
    else:
        canvas = image.copy()

    # Turn grayscale to rgb (if needed)
    if len(canvas.shape) == 2:
        canvas = np.stack([canvas] * 3, axis=2)
        
    # Draw all the non-zero values
    stroke_params = dict(color=(255, 0, 0), thickness=1, tipLength=.2)
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Get the flow vector at this point
            fx, fy = flow[y, x]

            # Calculate the magnitude of the flow vector
            M = np.sqrt(fx**2 + fy**2)
            
            if M > threshold:
                # Draw the arrowed line
                dx, dy = int(x + fx), int(y + fy)
                cv2.arrowedLine(canvas, (x, y), (dx, dy), **stroke_params)
            else:
                pass
    
    return canvas.astype(np.uint8)


def visualize_dense_color(flow: np.ndarray, color_model: str = 'BGR'):

    color_model = str(color_model).upper()

    # crate HSV & make Value a constant
    hsv = np.zeros(list(flow.shape[:2]) + [3]) 
    hsv[..., 1] = 255

    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Use Hue and Saturation to encode the Optical Flow
    hsv[..., 0] = ang * 180 / PI / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    if color_model == 'HSV':
        return hsv

    # Convert HSV image into BGR
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if color_model == 'BGR':
        return bgr

    # Revert order, BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def visualize_dense_flow(flow: np.ndarray, mode: str = 'color'):

    if mode == 'color':
        return visualize_dense_color(flow)

    elif mode == 'arrow':
        return visualize_dense_arrow(flow)

    else:
        raise ValueError(f'mode = {mode} is NOT supported!')


