from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml


def create_3d_chessboard_points(
    chessboard_size: Tuple, square_size: float
) -> np.ndarray:
    """Create 3D chessboard points for a chessboard with given size and square size.

    Parameters
    ----------
    chessboard_size : Tuple
        tuple of integers representing the number of squares in the chessboard
    square_size : float
        size of each square in the chessboard

    Returns
    -------
    np.ndarray
        Nx3 array of 3D points: N = number of squares in chessboard
    """
    # create Nx3 array of 3D points: N = number of squares in chessboard
    chessboard_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
    # fill x, y coordinates with indices of chessboard points
    chessboard_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    # scale coordinates by square size
    chessboard_points *= square_size
    return chessboard_points


def create_2d_chessboard_points(
    image: np.ndarray, chessboard_size: Tuple
) -> np.ndarray:
    """Create 2D chessboard points for a given image and chessboard size.

    Parameters
    ----------
    image : np.ndarray
        image of chessboard
    chessboard_size : Tuple
        tuple of integers representing the number of squares in the chessboard

    Returns
    -------
    np.ndarray
        Nx1x2 array of 2D points: N = number of corners found in the image
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # find chessboard corners
    found, corners = cv2.findChessboardCorners(image, chessboard_size, None)
    if not found:
        return None
    # refine corner positions
    corners = cv2.cornerSubPix(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria
    )
    return corners


def plot_chessboard(
    image: np.ndarray, corners: np.ndarray, chessboard_size: Tuple
) -> np.ndarray:
    """Plot chessboard corners on the image.

    Parameters
    ----------
    image : np.ndarray
        image of chessboard
    corners : np.ndarray
        Nx1x2 array of 2D points: N = number of corners found in the image
    chessboard_size : Tuple
        tuple of integers representing the number of squares in the chessboard

    Returns
    -------
    np.ndarray
        image with corners plotted
    """
    image = cv2.drawChessboardCorners(image, chessboard_size, corners, True)
    return image


def get_all_2d_points(images_path: Path, chessboard_size: Tuple) -> List[np.ndarray]:
    """Get all 2D chessboard points from images in a given directory.

    Parameters
    ----------
    images_path : Path
        path to directory containing images of chessboard
    chessboard_size : Tuple
        tuple of integers representing the number of squares in the chessboard

    Returns
    -------
    List[np.ndarray]
        list of Nx1x2 arrays of 2D points: N = number of corners found in each image
    """
    image_files = images_path.glob("*.png")
    all_corners = []
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corners = create_2d_chessboard_points(image, chessboard_size)
        if corners is not None:
            all_corners.append(corners)
    return all_corners


def calibrate_camera(
    chessboard_3d_points: List[np.array],
    chessboard_2d_points: List[np.array],
    image_size: Tuple,
) -> Tuple[bool, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Calibrate camera using 3D and 2D chessboard points.

    Parameters
    ----------
    chessboard_3d_points : List[np.array]
        list of Nx3 arrays of 3D points: N = number of squares in chessboard
    chessboard_2d_points : List[np.array]
        list of Nx1x2 arrays of 2D points: N = number of corners found in each image
    image_size : Tuple
        tuple of integers representing the image size

    Returns
    -------
    bool
        flag indicating if calibration was successful
    np.ndarray
        camera matrix
    np.ndarray
        distortion coefficients
    List[np.ndarray]
        list of rotation vectors
    List[np.ndarray]
        list of translation vectors
    """
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_3d_points, chessboard_2d_points, image_size, None, None
    )

    return ret, mtx, dist, rvecs, tvecs


def compute_reprojection_error(
    chessboard_3d_points: List[np.array],
    chessboard_2d_points: List[np.array],
    rvecs: List[np.array],
    tvecs: List[np.array],
    mtx: np.ndarray,
    dist: np.ndarray,
) -> float:
    """Compute reprojection error.

    Parameters
    ----------
    chessboard_3d_points : List[np.array]
        list of Nx3 arrays of 3D points: N = number of squares in chessboard
    chessboard_2d_points : List[np.array]
        list of Nx1x2 arrays of 2D points: N = number of corners found in each image
    rvecs : List[np.array]
        list of rotation vectors
    tvecs : List[np.array]
        list of translation vectors
    mtx : np.ndarray
        camera matrix
    dist : np.ndarray
        distortion coefficients

    Returns
    -------
    float
        mean reprojection error

    """
    mean_error = 0
    for i in range(len(chessboard_3d_points)):
        # project 3D points to image plane
        image_points, _ = cv2.projectPoints(
            chessboard_3d_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        # calculate error between projected points and detected points
        error = cv2.norm(chessboard_2d_points[i], image_points, cv2.NORM_L2) / len(
            image_points
        )
        mean_error += error
    mean_error /= len(chessboard_3d_points)
    return mean_error


def save_camera_parameters_as_yaml(
    yaml_file: str, mtx: np.ndarray, dist: np.ndarray
) -> None:
    """Save camera parameters to a yaml file.

    Parameters
    ----------
    yaml_file : str
        path to yaml file
    mtx : np.ndarray
        camera matrix  (3x3) intrinsic parameters
    dist : np.ndarray
        distortion coefficients (5x1) radial and tangential distortion coefficients
    """
    data = {"camera_matrix": mtx.tolist(), "distortion_coefficients": dist.tolist()}
    with open(yaml_file, "w") as file:
        yaml.dump(data, file)


def save_camera_parameters_as_xml(
    xml_file: str, mtx: np.ndarray, dist: np.ndarray
) -> None:
    """Save camera parameters to an xml file.

    Parameters
    ----------
    xml_file : str
        path to xml file
    mtx : np.ndarray
        camera matrix  (3x3) intrinsic parameters
    dist : np.ndarray
        distortion coefficients (5x1) radial and tangential distortion coefficients
    """

    cv_file = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("distortion_coefficients", dist)
    cv_file.release()


def read_camera_parameters(xml_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read camera parameters from an xml file.

    Parameters
    ----------
    xml_file : str
        path to xml file

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        camera matrix and distortion coefficients
    """
    cv_file = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("camera_matrix").mat()
    dist = cv_file.getNode("distortion_coefficients").mat()
    cv_file.release()
    return mtx, dist


def undistort_image(image: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort image using camera matrix and distortion coefficients.

    This function uses cv2.initUndistortRectifyMap and cv2.remap functions to
    undistort the image.

    Parameters
    ----------
    image : np.ndarray
        image to undistort
    mtx : np.ndarray
        camera matrix
    dist : np.ndarray
        distortion coefficients

    Returns
    -------
    np.ndarray
        undistorted image
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, new_camera_matrix, (w, h), 5
    )
    undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y : y + h, x : x + w]
    return undistorted_image


def simple_undistort_image(
    image: np.ndarray, mtx: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """Undistort image using camera matrix and distortion coefficients.

    This function uses cv2.undistort function to undistort the image.

    Parameters
    ----------
    image : np.ndarray
        image to undistort
    mtx : np.ndarray
        camera matrix
    dist : np.ndarray
        distortion coefficients

    Returns
    -------
    np.ndarray
        undistorted image
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, mtx, dist, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y : y + h, x : x + w]
    return undistorted_image


def plot_distorted_vs_undistorted_images(
    image: np.ndarray, undistorted_image: np.ndarray
) -> np.ndarray:
    """Stack the distorted and undistorted images side by side for comparison.

    Note that the images do not have the same size.

    Parameters
    ----------
    image : np.ndarray
        distorted image
    undistorted_image : np.ndarray
        undistorted image

    Returns
    -------
    np.ndarray
        stacked image
    """
    h, w = image.shape[:2]
    new_w = 2 * w
    new_h = h + 40
    stacked_image = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    stacked_image[:h, :w, :] = image
    h2, w2 = undistorted_image.shape[:2]
    stacked_image[:h2, w + (w - w2) :, :] = undistorted_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        stacked_image,
        "Distorted",
        (w // 2, h + 20),
        font,
        2,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        stacked_image,
        "Undistorted",
        (w + w // 2, h + 20),
        font,
        2,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return stacked_image
