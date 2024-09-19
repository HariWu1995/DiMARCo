"""
``rainymotion.models``: optical flow models for radar-based
precipitation nowcasting
===============================================================================

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

from .utils import Scaler, Scaler_inv
from .algorithms import *


class Sparse:
    """
        The basic class for the Sparse model of the rainymotion library.

        To run your nowcasting model you first have to set up a class instance as follows:

            `model = Sparse()`

        and then use class attributes to set up model parameters, e.g.:

            `model.extrapolation = "linear"`

        All class attributes have default values, for getting started with
        nowcasting you must specify only `input_data` attribute which holds
        the latest radar data observations. After specifying the input data,
        you can run nowcasting model and produce the corresponding results of
        nowcasting using `.run()` method:

        `nowcasts = model.run()`

        Attributes
        ----------
            input_data: 3D numpy array (frames, dim_x, dim_y) of radar data for
                        previous hours. "frames" dimension must be > 2.

            scaler: function, default=rainymotion.utils.Scaler
                Corner identification and optical flow algorithms require specific data
                type to perform calculations: uint8. That means that you must specify
                the transformation function (i.e. "scaler") to convert the "input_data"
                to the range of integers [0, 255]. 
                By default we are using Scaler which converts precipitation depth 
                (mm, float16) to "brightness" values (uint8).

            scaler_inv: function, default=rainymotion.utils.Scaler_inv
                Function which does the inverse transformation of "brightness"
                values (uint8) to precipitation values.

            lead_steps: int, default=12
                Number of lead times for which we want to produce nowcasts. Must be > 0

            of_params: dict
                The dictionary of corresponding Shi-Tomasi corner detector parameters
                (key "st_pars"), and Lukas-Kanade optical flow parameters
                (key "lk_pars"). The default dictionary for parameters is:
                {'st_pars' : dict(maxCorners = 200,
                                qualityLevel = 0.2,
                                minDistance = 7,
                                blockSize = 21 ),
                'lk_pars' : dict(winSize  = (20,20),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))}

            extrapolation: str, default="linear"
                The extrapolation method for precipitation features advection.
                Linear method establishes linear regression for every detected feature
                which then used to advect this feature to the imminent future.

            warper: str, default="affine", options=["affine", "euclidean", "similarity",
                                                    "projective"]
                    Warping technique used for transformation of the last available
                    radar observation in accordance with advected features displacement.

        Methods
        -------
            run(): perform calculation of nowcasts.
                Return 3D numpy array of shape (lead_steps, dim_x, dim_y).
    """
    def __init__(self, of_params=DEFAULT_PARAMS):

        self.warper = "affine"
        self.extrapolation = "linear"
        self.lead_steps = 12
        self.of_params = of_params

        self.scaler = Scaler
        self.scaler_inv = Scaler_inv

    def run(self, input_data):
        """
        Run nowcasting calculations.

        Returns
        -------
            nowcasts : 3D numpy array of shape (lead_steps, dim_x, dim_y).
        """
        # define available transformations dictionary
        transformations = {
                'affine':     sktf.AffineTransform(),
             'euclidean':  sktf.EuclideanTransform(),
            'similarity': sktf.SimilarityTransform(),
            'projective': sktf.ProjectiveTransform(),
        }

        # scale input data to uint8 [0-255] with self.scaler
        data_scaled, c1, c2 = self.scaler(input_data)

        # set up transformer object
        trf = transformations[self.warper]

        # obtain source and target points
        _sparse_func =  _sparse_sd if self.extrapolation == "simple_delta" else \
                        _sparse_linear
            
        pts_source, \
        pts_target_container = _sparse_func(data_instance=data_scaled,
                                            of_params=self.of_params,
                                            lead_steps=self.lead_steps)

        # now we can start to find nowcasted image
        # for every candidate of projected sets of points

        # container for our nowcasts
        last_frame = data_scaled[-1]
        nowcst_frames = []

        for lead_step, pts_target in enumerate(pts_target_container):

            # estimate transformation matrix based on source and traget points
            trf.estimate(pts_source, pts_target)

            # make a nowcast
            nowcst_frame = sktf.warp(last_frame/255, trf.inverse)

            # transformations dealing with strange behaviour
            nowcst_frame = (nowcst_frame*255).astype('uint8')

            # add to the container
            nowcst_frames.append(nowcst_frame)

        nowcst_frames = np.stack(nowcst_frames, axis=0)
        nowcst_frames = self.scaler_inv(nowcst_frames, c1, c2)

        return nowcst_frames


class SparseSD:
    """
        The basic class for the SparseSD model of the rainymotion library.

        To run your nowcasting model you first have to set up a class instance
        as follows:

            `model = SparseSD()`

        and then use class attributes to set up model parameters, e.g.:

            `model.warper = "affine"`

        All class attributes have default values, for getting started with
        nowcasting you must specify only `input_data` attribute which holds the
        latest radar data observations. After specifying the input data, you can
        run nowcasting model and produce the corresponding results of nowcasting
        using `.run()` method:

        `nowcasts = model.run()`

        Attributes
        ----------
        input_data: 3D numpy array (frames, dim_x, dim_y) of radar data for
                    previous hours. "frames" dimension must be > 2.

        scaler: function, default=rainymotion.utils.Scaler
            Corner identification and optical flow algorithms require specific data
            type to perform calculations: uint8. That means that you must specify
            the transformation function (i.e. "scaler") to convert the "input_data"
            to the range of integers [0, 255]. By default we are using Scaler
            which converts precipitation depth (mm, float16) to "brightness"
            values (uint8).

        scaler_inv: function, default=rainymotion.utils.Scaler_inv
            Function which does the inverse transformation of "brightness"
            values (uint8) to precipitation values.

        lead_steps: int, default=12
            Number of lead times for which we want to produce nowcasts. Must be > 0

        of_params: dict
            The dictionary of corresponding Shi-Tomasi corner detector parameters
            (key "st_pars"), and Lukas-Kanade optical flow parameters
            (key "lk_pars"). The default dictionary for parameters is:
            {'st_pars' : dict(maxCorners = 200,
                            qualityLevel = 0.2,
                            minDistance = 7,
                            blockSize = 21 ),
            'lk_pars' : dict(winSize  = (20,20),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))}

        extrapolation: str, default="simple_delta"
            The extrapolation method for precipitation features advection.
            For "simple_delta" method we use an assumption that detected
            displacement of precipitation feature between the two latest radar
            observations will be constant for each lead time.

        warper: str, default="affine", options=["affine", "euclidean", "similarity",
                                                "projective"]
            Warping technique used for transformation of the last available radar
            observation in accordance with advected features displacement.

        Methods
        -------
        run(): perform calculation of nowcasts.
            Return 3D numpy array of shape (lead_steps, dim_x, dim_y).
    """
    def __init__(self, of_params=DEFAULT_PARAMS):

        self.of_params = of_params
        self.lead_steps = 12

        self.warper = "affine"
        self.extrapolation = "simple_delta"

        self.scaler = Scaler
        self.scaler_inv = Scaler_inv

    def run(self, input_data):
        """
        Run nowcasting calculations.

        Returns
        -------
            nowcasts : 3D numpy array of shape (lead_steps, dim_x, dim_y).
        """
        # define available transformations dictionary
        transformations = {
                'affine':     sktf.AffineTransform(),
             'euclidean':  sktf.EuclideanTransform(),
            'similarity': sktf.SimilarityTransform(),
            'projective': sktf.ProjectiveTransform(),
        }

        # scale input data to uint8 [0-255] with self.scaler
        data_scaled, c1, c2 = self.scaler(input_data)

        # set up transformer object
        trf = transformations[self.warper]

        # obtain source and target points
        _sparse_func =  _sparse_sd
            
        pts_source, \
        pts_target_container = _sparse_func(data_instance=data_scaled,
                                            of_params=self.of_params,
                                            lead_steps=self.lead_steps)

        # now we can start to find nowcasted image
        # for every candidate of projected sets of points

        # container for our nowcasts
        last_frame = data_scaled[-1]
        nowcst_frames = []

        for lead_step, pts_target in enumerate(pts_target_container):

            # estimate transformation matrix based on source and traget points
            trf.estimate(pts_source, pts_target)

            # make a nowcast
            nowcst_frame = sktf.warp(last_frame/255, trf.inverse)

            # transformations dealing with strange behaviour
            nowcst_frame = (nowcst_frame*255).astype('uint8')

            # add to the container
            nowcst_frames.append(nowcst_frame)

        nowcst_frames = np.stack(nowcst_frames, axis=0)
        nowcst_frames = self.scaler_inv(nowcst_frames, c1, c2)

        return nowcst_frames


class Dense:
    """
        The basic class for the Dense model of the rainymotion library.

        To run your nowcasting model you first have to set up a class instance
        as follows:

        `model = Dense()`

        and then use class attributes to set up model parameters, e.g.:

        `model.of_method = "Farneback"`

        All class attributes have default values, for getting started with
        nowcasting you must specify only `input_data` attribute which holds the
        latest radar data observations. After specifying the input data, you can
        run nowcasting model and produce the corresponding results of nowcasting
        using `.run()` method:

        `nowcasts = model.run()`

        Attributes
        ----------
        input_data: 3D numpy array (frames, dim_x, dim_y) of radar data for
        previous hours. "frames" dimension must be > 2.

        scaler: function, default=rainymotion.utils.Scaler
            Corner identification and optical flow algorithms require specific data
            type to perform calculations: uint8. That means that you must specify
            the transformation function (i.e. "scaler") to convert the "input_data"
            to the range of integers [0, 255]. By default we are using Scaler
            which converts precipitation depth (mm, float16) to "brightness"
            values (uint8).

        lead_steps: int, default=12
            Number of lead times for which we want to produce nowcasts. Must be > 0

        of_method: str, default="Farneback", 
                        options=["DIS", "PCAFlow", "DeepFlow", "Farneback"]
            The optical flow method to obtain the dense representation (in every
            image pixel) of motion field. By default we use the Dense Inverse
            Search algorithm (DIS). PCAFlow, DeepFlow, and Farneback algoritms
            are also available to obtain motion field.

        advection: str, default="constant-vector"
            The advection scheme we use for extrapolation of every image pixel
            into the imminent future.

        direction: str, default="backward", options=["forward", "backward"]
            The direction option of the advection scheme.

        interpolation: str, default="idw", options=["idw", "nearest", "linear"]
            The interpolation method we use to interpolate advected pixel values
            to the original grid of the radar image. By default we use inverse
            distance weightning interpolation (Idw) as proposed in library wradlib
            (wradlib.ipol.Idw), but interpolation techniques from scipy.interpolate
            (e.g., "nearest" or "linear") could also be used.

        Methods
        -------
        run(): perform calculation of nowcasts.
            Return 3D numpy array of shape (lead_steps, dim_x, dim_y).
    """

    def __init__(self):

        self.scaler = Scaler
        self.lead_steps = 12

        self.of_method = "Farneback"
        self.direction = "backward"
        self.advection = "constant-vector"
        self.interpolation = "idw"

    def run(self, input_data):
        """
        Run nowcasting calculations.

        Returns
        -------
            nowcasts : 3D numpy array of shape (lead_steps, dim_x, dim_y).
        """
        # scale input data to uint8 [0-255] with self.scaler
        scaled_data, c1, c2 = self.scaler(input_data)

        # calculate optical flow
        of = _calculate_optical_flow(scaled_data, method=self.of_method, direction=self.direction)

        # advect pixels accordingly
        _advection_func = _advection_semi_lagrangian if self.advection == "semi-lagrangian" else \
                          _advection_constant_vector
        coord_source, \
        coord_targets = _advection_func(of, lead_steps=self.lead_steps)

        # nowcasts placeholder
        nowcasts = []

        # interpolation
        for lead_step in range(self.lead_steps):
            nowcasts.append(
                  _interpolator(input_data[-1], 
                                coord_source,
                                coord_targets[lead_step], 
                                method=self.interpolation))

        # reshaping
        nowcasts = np.moveaxis(np.dstack(nowcasts), -1, 0)

        return nowcasts


class DenseRotation:
    """
        The basic class for the Dense model of the rainymotion library.

        To run your nowcasting model you first have to set up a class instance
        as follows:

        `model = Dense()`

        and then use class attributes to set up model parameters, e.g.:

        `model.of_method = "Farneback"`

        All class attributes have default values, for getting started with
        nowcasting you must specify only `input_data` attribute which holds the
        latest radar data observations. After specifying the input data, you can
        run nowcasting model and produce the corresponding results of nowcasting
        using `.run()` method:

        `nowcasts = model.run()`

        Attributes
        ----------
        input_data: 3D numpy array (frames, dim_x, dim_y) of radar data for
                    previous hours. "frames" dimension must be > 2.

        scaler: function, default=rainymotion.utils.Scaler
            Corner identification and optical flow algorithms require specific data
            type to perform calculations: uint8. That means that you must specify
            the transformation function (i.e. "scaler") to convert the "input_data"
            to the range of integers [0, 255]. By default we are using Scaler
            which converts precipitation depth (mm, float16) to "brightness"
            values (uint8).

        lead_steps: int, default=12
            Number of lead times for which we want to produce nowcasts. Must be > 0

        of_method: str, default="Farneback", 
                        options=["DIS", "PCAFlow", "DeepFlow", "Farneback"]
            The optical flow method to obtain the dense representation (in every
            image pixel) of motion field. By default we use the Dense Inverse
            Search algorithm (DIS). PCAFlow, DeepFlow, and Farneback algoritms
            are also available to obtain motion field.

        advection: str, default="semi-lagrangian"
            The advection scheme we use for extrapolation of every image pixel
            into the imminent future.

        direction: str, default="backward", options=["forward", "backward"]
            The direction option of the advection scheme.

        interpolation: str, default="idw", options=["idw", "nearest", "linear"]
            The interpolation method we use to interpolate advected pixel values
            to the original grid of the radar image. By default we use inverse
            distance weightning interpolation (idw) as proposed in wradlib.ipol.Idw
            but interpolation techniques from scipy.interpolate (e.g., "nearest"
            or "linear") could also be used.

        Methods
        -------
        run(): perform calculation of nowcasts.
            Return 3D numpy array of shape (lead_steps, dim_x, dim_y).
    """
    def __init__(self):

        self.scaler = Scaler

        self.lead_steps = 12

        self.of_method = "Farneback"
        self.direction = "backward"
        self.advection = "semi-lagrangian"
        self.interpolation = "idw"

    def run(self, input_data):
        """
        Run nowcasting calculations.

        Returns
        -------
            nowcasts : 3D numpy array of shape (lead_steps, dim_x, dim_y).
        """
        # scale input data to uint8 [0-255] with self.scaler
        scaled_data, c1, c2 = self.scaler(input_data)

        # calculate optical flow
        of = _calculate_optical_flow(scaled_data, method=self.of_method, direction=self.direction)

        # advect pixels accordingly
        _advection_func = _advection_semi_lagrangian if self.advection == "semi-lagrangian" else \
                          _advection_constant_vector    
        coord_source, \
        coord_targets = _advection_func(of, lead_steps=self.lead_steps)

        # nowcasts placeholder
        nowcasts = []

        # interpolation
        for lead_step in range(self.lead_steps):
            nowcasts.append(
                  _interpolator(input_data[-1], 
                                coord_source,
                                coord_targets[lead_step],
                                method=self.interpolation)
            )

        # reshaping
        nowcasts = np.moveaxis(np.dstack(nowcasts), -1, 0)

        return nowcasts


class Persistence:

    """
        The basic class of the Eulerian Persistence model (Persistence)
        of the rainymotion library.

        To run your nowcasting model you first have to set up a class instance
        as follows:

        `model = Persistence()`

        and then use class attributes to set up model parameters, e.g.:

        `model.lead_steps = 12`

        For getting started with nowcasting you must specify only `input_data`
        attribute which holds the latest radar data observations.
        After specifying the input data, you can run nowcasting model and
        produce the corresponding results of nowcasting using `.run()` method:

        `nowcasts = model.run()`

        Attributes
        ----------

        input_data: 3D numpy array (frames, dim_x, dim_y) of radar data for
                    previous hours. "frames" dimension must be > 2.

        lead_steps: int, default=12
            Number of lead times for which we want to produce nowcasts. Must be > 0

        Methods
        -------
        run(): perform calculation of nowcasts.
            Return 3D numpy array of shape (lead_steps, dim_x, dim_y).
    """

    def __init__(self):
        self.lead_steps = 12

    def run(self, input_data):
        """
        Run nowcasting calculations.

        Returns
        -------
            nowcasts : 3D numpy array of shape (lead_steps, dim_x, dim_y).
        """
        last_frame = input_data[-1, :, :]

        forecast = np.dstack([last_frame for i in range(self.lead_steps)])

        return np.moveaxis(forecast, -1, 0).copy()
