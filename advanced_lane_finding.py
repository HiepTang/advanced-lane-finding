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

    def __init__(self,
                 chessboard_image_dir='chessboard_images',
                 absolute_sobel_x=(7, 15, 100),
                 absolute_sobel_y=(7, 15, 100),
                 magnitude_sobel=(7, 30, 100),
                 direction_sobel=(31, 0.5, 1.0),
                 s_channel_thresh=(170, 255),
                 warp_perspective=(np.float32([(288, 664),
                                               (1016, 664),
                                               (240, 696),
                                               (1064, 696)]),
                                   np.float32([(240, 640),
                                               (1064, 640),
                                               (240, 696),
                                               (1064, 696)]))
                 ) -> None:
        """Initialize AdvancedLaneFinder instance fields.

        Args:
            :param chessboard_image_dir: path to directory containig chessboard calibration images
            :param absolute_sobel_x:  tuple containing Sobel_x kernel size, min and max threshold values for
                                      absolute value of Sobel_x operator
            :param absolute_sobel_y:  tuple containing Sobel_y kernel size, min and max threshold values for
                                      absolute value of Sobel_y operator
            :param magnitude_sobel: tuple containing Sobel kernel size, min and max threshold values for
                                    magnitude value of Sobel_x and Sobel_y operators
            :param direction_sobel: tuple containing Sobel kernel size, min and max threshold values for
                                    direction value of Sobel_x and Sobel_y operators
            :param s_channel_thresh: tuple containing min and max threshold values fof S channel of HLS image
            :param warp_perspective: tuple containing source and destination coordinates
                                     to calculate a perspective transform
        """
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

        # image size, calibration matrix, distortion coefficients, rotation vectors, and translation vectors
        # will be initialized in self.calibrate_camera()
        self._calibration_matrix = None
        self._distortion_coefficients = None
        self._rotation_vectors = None
        self._translation_vectors = None
        self._image_size = None

        # thresholds (absolute Sobel_x)
        if absolute_sobel_x[1] > absolute_sobel_x[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for absolute Sobel_x operator are incorrect; minimum greater than maximum [ %s ]."
                % absolute_sobel_x)
        self._abs_sobel_x_kernel = absolute_sobel_x[0]
        self._abs_sobel_x_thresh_min = absolute_sobel_x[1]
        self._abs_sobel_x_thresh_max = absolute_sobel_x[2]

        # thresholds (absolute Sobel_y)
        if absolute_sobel_y[1] > absolute_sobel_y[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for absolute Sobel_y operator are incorrect; minimum greater than maximum [ %s ]."
                % absolute_sobel_y)
        self._abs_sobel_y_kernel = absolute_sobel_y[0]
        self._abs_sobel_y_thresh_min = absolute_sobel_y[1]
        self._abs_sobel_y_thresh_max = absolute_sobel_y[2]

        # thresholds (magnitude Sobel)
        if magnitude_sobel[1] > magnitude_sobel[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for magnitude Sobel x and y operators are incorrect; minimum greater than maximum [ %s ]."
                % magnitude_sobel)
        self._mag_sobel_kernel = magnitude_sobel[0]
        self._mag_sobel_thresh_min = magnitude_sobel[1]
        self._mag_sobel_thresh_max = magnitude_sobel[2]

        # thresholds (direction Sobel)
        if direction_sobel[1] > direction_sobel[2]:
            raise AdvancedLaneFinderError(
                "Thresholds for direction Sobel x and y operators are incorrect; minimum greater than maximum [ %s ]."
                % direction_sobel)
        self._dir_sobel_kernel = direction_sobel[0]
        self._dir_sobel_thresh_min = direction_sobel[1]
        self._dir_sobel_thresh_max = direction_sobel[2]

        # thresholds (S channel of HLS)
        if s_channel_thresh[0] > s_channel_thresh[1]:
            raise AdvancedLaneFinderError(
                "Thresholds for S channel of HLS image are incorrect; minimum greater than maximum [ %s ]."
                % s_channel_thresh)
        self._s_channel_thresh_min = s_channel_thresh[0]
        self._s_channel_thresh_max = s_channel_thresh[1]

        # source and destination coordinates of quadrangle vertices
        self._warp_src_vertices = warp_perspective[0]
        self._warp_dst_vertices = warp_perspective[1]
        # calculate the perspective transform matrix
        self._perspective_transform_matrix = \
            cv2.getPerspectiveTransform(self._warp_src_vertices, self._warp_dst_vertices)

    def get_chessboard_image_list(self) -> List[str]:
        """Getter for chessboard image path list."""
        return self._chessboard_image_path_list

    def get_calibration_camera_output(self) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """Getter for the tuple of calibration matrix, distortion coefficients, rotation and translation vectors."""
        return (self._calibration_matrix,
                self._distortion_coefficients,
                self._rotation_vectors,
                self._translation_vectors)

    def calibrate_camera(self) -> Tuple[np.ndarray, np.ndarray, list, list]:
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

        # initialize image size
        self._image_size = img_shape

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

    def distortion_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply distortion correction to the input image."""
        return cv2.undistort(image,
                             self._calibration_matrix,
                             self._distortion_coefficients,
                             None,
                             self._calibration_matrix)

    def warp_perspective(self, image):
        """Calculates a perspective transform from four pairs of the corresponding points."""
        # warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(image, self._perspective_transform_matrix, self._image_size)

        return warped

    def _dir_threshold(self, gray) -> np.ndarray:
        """Apply threshold to gray scale image using direction of the gradient."""
        # calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._dir_sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._dir_sobel_kernel)

        # take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= self._dir_sobel_thresh_min) & (absgraddir <= self._dir_sobel_thresh_max)] = 1
        return binary_output

    def _mag_threshold(self, gray) -> np.ndarray:
        """Apply threshold to gray scale image using magnitude of the gradient."""
        # take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._mag_sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._mag_sobel_kernel)

        # calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)

        # rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)

        # create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self._mag_sobel_thresh_min) & (gradmag <= self._mag_sobel_thresh_max)] = 1
        return binary_output

    def _abs_threshold(self, gray, orient) -> np.ndarray:
        """Apply threshold to gray scale image using absolute value of the gradient for either Sobel_x or Sobel_y."""
        # apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._abs_sobel_x_kernel))
            thresh_min = self._abs_sobel_x_thresh_min
            thresh_max = self._abs_sobel_x_thresh_max
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._abs_sobel_y_kernel))
            thresh_min = self._abs_sobel_y_thresh_min
            thresh_max = self._abs_sobel_y_thresh_max
        else:
            raise AdvancedLaneFinderError("'orient' parameter of self._abs_thresh should be either 'x' or 'y'.")
        # rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output

    def _s_channel_threshold(self, hls):
        s_channel = hls[:, :, 2]  # use S channel

        # create a copy and apply the threshold
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel >= self._s_channel_thresh_min) & (s_channel <= self._s_channel_thresh_max)] = 1
        return binary_output

    def apply_thresholds(self, image: np.ndarray) -> np.ndarray:
        """Create a thresholded binary image."""
        # convert to HLS
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        # gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # binary images
        abs_x_binary = self._abs_threshold(gray, orient='x')
        abs_y_binary = self._abs_threshold(gray, orient='y')
        mag_binary = self._mag_threshold(gray)
        dir_binary = self._dir_threshold(gray)
        s_channel_binary = self._s_channel_threshold(hls)

        # combine thresholded images
        combined = np.zeros_like(dir_binary)
        combined[((abs_x_binary == 1) & (abs_y_binary == 1))
                 | ((mag_binary == 1) & (dir_binary == 1))
                 | (s_channel_binary == 1)] = 1
        return combined
