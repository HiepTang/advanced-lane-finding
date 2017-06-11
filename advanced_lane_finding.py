import os
import logging
import numpy as np
import cv2

from typing import Tuple, List


class AdvancedLaneFinderError(Exception):
    """Exception to be thrown in case of failure in AdvancedLaneFinding."""
    pass


class AdvancedLaneFinder:
    """Class implementing lane line finding using advanced techniques."""

    def __init__(self, chessboard_image_dir) -> None:
        """Initialize AdvancedLaneFinder instance fields."""
        # initialize directory with chessboard images
        if os.path.isdir(chessboard_image_dir):
            self._chessboard_image_dir = os.path.abspath(chessboard_image_dir)
            logging.info("Directory with calibration images: %s.", self._chessboard_image_dir)
        else:
            raise AdvancedLaneFinderError("%s directory does not exist." % chessboard_image_dir)

        # initialize list of calibration chessboard images
        self._chessboard_image_path_list = [os.path.join(self._chessboard_image_dir, fname)
                                            for fname in os.listdir(self._chessboard_image_dir)]
        if not self._chessboard_image_path_list:
            raise AdvancedLaneFinderError("No calibration images found in %s." % self._chessboard_image_dir)
        else:
            logging.info("There are %d calibration images.", len(self._chessboard_image_path_list))

        # calibration matrix, distortion coefficients, rotation vectors, and translation vectors
        # will be initialized in self.calibrate_camera()
        self._calibration_matrix = None
        self._distortion_coefficients = None
        self._rotation_vectors = None
        self._translation_vectors = None

    def get_chessboard_image_list(self) -> List[str]:
        """Getter for chessboard image path list."""
        return self._chessboard_image_path_list

    def get_calibration_camera_output(self):
        """Getter for the tuple of calibration matrix, distortion coefficients, rotation and translation vectors."""
        return (self._calibration_matrix,
                self._distortion_coefficients,
                self._rotation_vectors,
                self._translation_vectors)

    def calibrate_camera(self) -> Tuple:
        """Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

        Return tuple of calibration matrix and distortion coefficients.
        """

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # get reference image shape
        img_shape = cv2.imread(self._chessboard_image_path_list[0]).shape[1::-1]

        # step through the list of chessboard image paths and search for chessboard corners
        for fname in self._chessboard_image_path_list:
            img = cv2.imread(fname)

            # images should have equal shape
            assert img_shape == img.shape[1::-1]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # if found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        # calibrate camera; all object and image points are already collected
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=objpoints,
                                                           imagePoints=imgpoints,
                                                           imageSize=img_shape,
                                                           cameraMatrix=None,
                                                           distCoeffs=None)

        if not ret:
            raise AdvancedLaneFinderError("Camera calibration has failed.")
        else:
            # initialize corresponding class instance fields
            self._calibration_matrix = mtx
            self._distortion_coefficients = dist
            self._rotation_vectors = rvecs
            self._translation_vectors = tvecs

            return self.get_calibration_camera_output()

    def distortion_correction(self, image):
        """Apply distortion correction to the input image."""
        return cv2.undistort(image,
                             self._calibration_matrix,
                             self._distortion_coefficients,
                             None,
                             self._calibration_matrix)
