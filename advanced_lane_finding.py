import os
import logging

from typing import Tuple, List


class AdvancedLaneFinderError(Exception):
    """Exception to be thrown in case of failure in AdvancedLaneFinding."""
    pass


class AdvancedLaneFinder:
    """Class implementing lane line finding using advanced techniques."""

    def __init__(self, chessboard_image_dir) -> None:
        # initialize directory with chessboard images
        if os.path.isdir(chessboard_image_dir):
            self._chessboard_image_dir = os.path.abspath(chessboard_image_dir)
            logging.info("Directory with calibration images: %s", self._chessboard_image_dir)
        else:
            raise AdvancedLaneFinderError("%s directory does not exist" % chessboard_image_dir)

        # initialize list of calibration chessboard images
        self._chessboard_image_path_list = [os.path.join(self._chessboard_image_dir, fname)
                                            for fname in os.listdir(self._chessboard_image_dir)]
        if not self._chessboard_image_path_list:
            raise AdvancedLaneFinderError("No calibration images found in %s" % self._chessboard_image_dir)
        else:
            logging.info("There are %d calibration images", len(self._chessboard_image_path_list))

    def get_chessboard_image_list(self) -> List[str]:
        """Getter for chessboard image path list."""
        return self._chessboard_image_path_list

    def calibrate_camera(self) -> Tuple:
        """Compute the camera calibration matrix and distortion coefficients given a set of chessboard images."""
        import numpy as np
        import cv2
        import glob
        import matplotlib.pyplot as plt

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('../camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()
