from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from utils import (
    create_3d_chessboard_points,
    plot_chessboard,
    get_all_2d_points,
    calibrate_camera,
    compute_reprojection_error,
    save_camera_parameters_as_yaml,
    save_camera_parameters_as_xml,
    read_camera_parameters,
    undistort_image,
    simple_undistort_image,
    plot_distorted_vs_undistorted_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument(
        "--chessboard_size",
        type=str,
        default="24,17",
        help="Chessboard size (width, height)",
    )
    parser.add_argument(
        "--square_size", type=int, default=15, help="Chessboard square size (mm)"
    )
    return parser.parse_args()


def main(chessboard_size: Tuple[int, int] = (25, 18), square_size: int = 15) -> None:
    # Create 2D chessboard points
    images_path = Path("data/images/")
    chessboard_2d_points = get_all_2d_points(images_path, chessboard_size)

    # Plot chessboard corners
    for image_file, corners in zip(images_path.glob("*.png"), chessboard_2d_points):
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = plot_chessboard(image, corners, chessboard_size)
        cv2.imwrite(
            f"data/corners/{image_file.stem}_corners.png",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

    # Create 3D chessboard points
    chessboard_3d_points = create_3d_chessboard_points(chessboard_size, square_size)
    chessboard_3d_points = [chessboard_3d_points] * len(chessboard_2d_points)

    # Get image size
    image = cv2.imread(str(images_path.glob("*.png").__next__()))
    image_size = image.shape[:2][::-1]

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        chessboard_3d_points, chessboard_2d_points, image_size
    )

    # Compute reprojection error
    error = compute_reprojection_error(
        chessboard_3d_points, chessboard_2d_points, rvecs, tvecs, mtx, dist
    )
    print("Reprojection error:", error)

    # Save camera parameters
    save_camera_parameters_as_yaml("data/params/camera_parameters.yaml", mtx, dist)
    save_camera_parameters_as_xml("data/params/camera_parameters.xml", mtx, dist)

    # Read camera parameters
    mtx, dist = read_camera_parameters("data/params/camera_parameters.xml")

    # Undistort image
    image = cv2.imread(str(images_path.glob("*.png").__next__()))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted_image = undistort_image(image, mtx, dist)
    cv2.imwrite(
        "data/undistorted/undistorted_image.png",
        cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR),
    )

    # Plot distorted vs undistorted images
    viz = plot_distorted_vs_undistorted_images(image, undistorted_image)
    cv2.imwrite(
        "data/undistorted/distorted_vs_undistorted_images.png",
        cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
    )

    # Simple undistort image
    image = cv2.imread(str(images_path.glob("*.png").__next__()))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted_image = simple_undistort_image(image, mtx, dist)
    cv2.imwrite(
        "data/undistorted/simple_undistorted_image.png",
        cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR),
    )


if __name__ == "__main__":
    args = parse_args()
    chessboard_size = tuple(map(int, args.chessboard_size.split(",")))
    main(chessboard_size, args.square_size)
