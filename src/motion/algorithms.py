import numpy as np
import cv2

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import skimage.transform as sktf

import wradlib.ipol as ipol


DEFAULT_PARAMS = {
    'st_pars': dict(maxCorners = 200,
                    qualityLevel = 0.2,
                    minDistance = 7,
                    blockSize = 21),
    
    'lk_pars': dict(winSize = (20, 20),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0)),
}


# -- SPARSE GROUP -- #

def _sparse_linear(data_instance,
                   of_params=DEFAULT_PARAMS,
             extrapol_params={"model": LinearRegression(),
                              "features": "ordinal"},
                   lead_steps: int=12):

    # find features to track
    old_corners = cv2.goodFeaturesToTrack(data_instance[0], mask=None, **of_params['st_pars'])

    # Set containers to collect results (time steps in rows, detected corners in columns)

    # corner xy coords
    x = np.full((data_instance.shape[0], len(old_corners)), np.nan)
    y = np.full((data_instance.shape[0], len(old_corners)), np.nan)

    # Assign persistent corner IDs
    ids = np.arange(len(old_corners))

    # fill in first values
    x[0, :] = old_corners[:, 0, 0]
    y[0, :] = old_corners[:, 0, 1]

    # track corners by optical flow algorithm
    for i in range(1, data_instance.shape[0]):

        new_corners, st, err = cv2.calcOpticalFlowPyrLK(prevImg=data_instance[i-1],
                                                        nextImg=data_instance[i],
                                                        prevPts=old_corners,
                                                        nextPts=None,
                                                        **of_params['lk_pars'])

        # select only good attempts for corner tracking
        success = st.ravel() == 1

        # use only sucessfull ids for filling
        ids = ids[success]

        # fill in results
        x[i, ids] = new_corners[success, 0, 0]
        y[i, ids] = new_corners[success, 0, 1]

        # new corners will be old in the next loop
        old_corners = new_corners[success]

    # consider only full paths
    full_paths_without_nan = [np.sum(np.isnan(x[:, i])) == 0 for i in range(x.shape[1])]
    x = x[:, full_paths_without_nan].copy()
    y = y[:, full_paths_without_nan].copy()

    # containers for corners predictions
    x_new = np.full((lead_steps, x.shape[1]), np.nan)
    y_new = np.full((lead_steps, y.shape[1]), np.nan)

    for i in range(x.shape[1]):

        x_train = x[:, i]
        y_train = y[:, i]

        X = np.arange(x.shape[0] + lead_steps)

        if extrapol_params["features"] == "polynomial":
            polyfeatures = PolynomialFeatures(2)
            X = polyfeatures.fit_transform(X.reshape(-1, 1))
            X_train = X[:x.shape[0], :]
            X_pred = X[x.shape[0]:, :]
            
        else:
            X = X.reshape(-1, 1)
            X_train = X[:x.shape[0], :]
            X_pred = X[x.shape[0]:, :]

        x_pred = extrapol_params["model"].fit(X_train, x_train).predict(X_pred)
        y_pred = extrapol_params["model"].fit(X_train, y_train).predict(X_pred)

        x_new[:, i] = x_pred
        y_new[:, i] = y_pred

    # define source corners in appropriate format
    pts_source = np.hstack([x[-1, :].reshape(-1, 1), y[-1, :].reshape(-1, 1)])

    # define container for targets in appropriate format
    pts_target_container = [np.hstack([x_new[i, :].reshape(-1, 1),
                                       y_new[i, :].reshape(-1, 1)]) for i in range(x_new.shape[0])]

    return pts_source, pts_target_container


def _sparse_sd(data_instance, of_params=DEFAULT_PARAMS, lead_steps=12):

    # define penult and last frames
    penult_frame = data_instance[-2]
    last_frame = data_instance[-1]

    # find features to track
    old_corners = cv2.goodFeaturesToTrack(data_instance[0], mask=None, **of_params['st_pars'])

    # track corners by optical flow algorithm
    new_corners, stt, _ = cv2.calcOpticalFlowPyrLK( prevImg=penult_frame,
                                                    nextImg=last_frame,
                                                    prevPts=old_corners,
                                                    nextPts=None,
                                                    **of_params['lk_pars'])

    # select only good attempts for corner tracking
    success = stt.ravel() == 1
    new_corners = new_corners[success].copy()
    old_corners = old_corners[success].copy()

    # calculate Simple-Delta
    delta = new_corners.reshape(-1, 2) - old_corners.reshape(-1, 2)

    # simplificate further transformations
    pts_source = new_corners.reshape(-1, 2)

    # propagate our corners through time
    pts_target_container = []
    for lead_step in range(lead_steps):
        pts_target_container.append(pts_source + delta * (lead_step + 1))

    return pts_source, pts_target_container


# -- DENSE GROUP -- #

# filling holes (zeros) in velocity field
def _fill_holes(of_instance, threshold=0):

    # calculate velocity scalar
    vlcty = np.sqrt(of_instance[::, ::, 0] ** 2 + of_instance[::, ::, 1] ** 2)

    # zero mask
    zero_holes = vlcty <= threshold

    # targets
    coord_target_i, coord_target_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))

    # source
    coord_source_i = coord_target_i[~zero_holes]
    coord_source_j = coord_target_j[~zero_holes]

    delta_x_source = of_instance[::, ::, 0][~zero_holes]
    delta_y_source = of_instance[::, ::, 1][~zero_holes]

    # reshape
    src = np.vstack((coord_source_i.ravel(), coord_source_j.ravel())).T
    trg = np.vstack((coord_target_i.ravel(), coord_target_j.ravel())).T

    # create an object
    interpolator = ipol.Idw(src, trg)

    #
    delta_x_target = interpolator(delta_x_source.ravel())
    delta_y_target = interpolator(delta_y_source.ravel())

    # reshape output
    delta_x_target = delta_x_target.reshape(of_instance.shape[0], of_instance.shape[1])
    delta_y_target = delta_y_target.reshape(of_instance.shape[0], of_instance.shape[1])

    return np.stack([delta_x_target, delta_y_target], axis=-1)


# calculate optical flow
def _calculate_optical_flow(data_instance, method="Farneback", direction="forward"):

    data_instance = data_instance[-2:]

    # define frames order
    if direction == "forward":
        prev_frame = data_instance[0]
        next_frame = data_instance[1]
        coef = 1.0

    elif direction == "backward":
        prev_frame = data_instance[1]
        next_frame = data_instance[0]
        coef = -1.0

    # calculate dense flow
    of_params = [None]

    if method == "Farneback":
        # of_instance = cv2.optflow.createOptFlow_Farneback()
        of_function = cv2.calcOpticalFlowFarneback
        of_params.extend([0.5, 3, 15, 3, 5, 1.2, 0])

    # elif method == "DIS":
    #     # of_instance = cv2.optflow.createOptFlow_DIS()
    #     of_function = 

    # elif method == "DeepFlow":
    #     # of_instance = cv2.optflow.createOptFlow_DeepFlow()
    #     of_function = 

    # elif method == "PCAFlow":
    #     # of_instance = cv2.optflow.createOptFlow_PCAFlow()
    #     of_function = 

    elif method == "SimpleFlow":
        # of_instance = cv2.optflow.createOptFlow_SimpleFlow()
        of_function = cv2.optflow.calcOpticalFlowSF
        of_params = dict(layers=3, averaging_block_size=2, max_flow=2)
        of_params = list(of_params.values())

    elif method == "SparseToDense":
        # of_instance = cv2.optflow.createOptFlow_SparseToDense()
        of_function = cv2.optflow.calcOpticalFlowSparseToDense

    elif method == "RobustLocal":
        # of_instance = cv2.optflow.createOptFlow_DenseRLOF()
        of_function = cv2.optflow.calcOpticalFlowDenseRLOF

    # delta = of_instance.calc(prev_frame, next_frame, None) * coef
    delta = of_function(prev_frame, next_frame, *of_params) * coef

    # variational refinement
    if method in ["Farneback", "SimpleFlow"]:
        try:
            refiner = cv2.optflow.createVariationalFlowRefinement()
            delta = refiner.calc(prev_frame, next_frame, delta)
        except Exception as e:
            print('Ignore Refining step due to:\n', e, '\n\n')

        delta = np.nan_to_num(delta)
        delta = _fill_holes(delta)

    return delta


# constant-vector advection
def _advection_constant_vector(of_instance, lead_steps=12):

    delta_x = of_instance[::, ::, 0]
    delta_y = of_instance[::, ::, 1]

    # make a source meshgrid
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))

    # calculate new coordinates of radar pixels
    coord_targets = []
    for lead_step in range(lead_steps):
        coord_target_i = coord_source_i + delta_x * (lead_step + 1)
        coord_target_j = coord_source_j + delta_y * (lead_step + 1)
        coord_targets.append([coord_target_i, coord_target_j])

    coord_source = [coord_source_i, coord_source_j]

    return coord_source, coord_targets


# semi-Lagrangian advection
def _advection_semi_lagrangian(of_instance, lead_steps=12):

    delta_x = of_instance[::, ::, 0]
    delta_y = of_instance[::, ::, 1]

    # make a source meshgrid
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))

    # create dynamic delta holders
    delta_xi = delta_x.copy()
    delta_yi = delta_y.copy()

    # Block for calculation displacement
    # init placeholders
    coord_targets = []
    for lead_step in range(lead_steps):

        # calculate corresponding targets
        coord_target_i = coord_source_i + delta_xi
        coord_target_j = coord_source_j + delta_yi
        coord_targets.append([coord_target_i, coord_target_j])

        # now update source coordinates
        coord_source_i = coord_target_i
        coord_source_j = coord_target_j
        coord_source = [coord_source_j.ravel(), coord_source_i.ravel()]

        # update deltas
        delta_xi = map_coordinates(delta_x, coord_source).reshape(of_instance.shape[0], of_instance.shape[1])
        delta_yi = map_coordinates(delta_y, coord_source).reshape(of_instance.shape[0], of_instance.shape[1])

    # reinitialization of coordinates source
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))
    coord_source = [coord_source_i, coord_source_j]

    return coord_source, coord_targets


# interpolation routine
def _interpolator(points, coord_source, coord_target, method="idw"):

    coord_source_i, coord_source_j = coord_source
    coord_target_i, coord_target_j = coord_target

    # reshape
    trg = np.vstack((coord_source_i.ravel(), coord_source_j.ravel())).T
    src = np.vstack((coord_target_i.ravel(), coord_target_j.ravel())).T

    if method == "nearest":
        interpolator = NearestNDInterpolator(src, points.ravel(), tree_options={"balanced_tree": False})
        points_interpolated = interpolator(trg)

    elif method == "linear":
        interpolator = LinearNDInterpolator(src, points.ravel(), fill_value=0)
        points_interpolated = interpolator(trg)

    elif method == "idw":
        interpolator = ipol.Idw(src, trg)
        points_interpolated = interpolator(points.ravel())

    # reshape output
    points_interpolated = points_interpolated.reshape(points.shape)

    return points_interpolated.astype(points.dtype)



