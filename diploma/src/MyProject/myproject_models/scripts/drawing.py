import cv2
import numpy as np


def draw_keypoints_and_limbs_for_one_person(img: np.ndarray, keypoints: list) -> np.ndarray:
    """Function drawing keypoints and limbs to target image

    Args:
        img (np.ndarray): target image
        keypoints (list): keypoints

    Returns:
        np.ndarray: output image
    """
    # создаём копию изображений
    img_copy = img.copy()
    point_color = (0, 255, 0)
    limb_color = (0, 0, 255)
    limbs = [[2, 0], [2, 4], [1, 0], [1, 3], [6, 8], [8, 10], [5, 7], [7, 9], [12, 14], 
             [14, 16], [11, 13], [13, 15], [6, 5], [12, 11], [6, 12], [5, 11]]
    for keypoint in keypoints:
        # рисуем кружок радиуса 5 вокруг точки
        cv2.circle(img_copy, tuple(keypoint), 5, point_color, -1)
    for limb in limbs:
        point0 = tuple(keypoints[limb[0]])
        point1 = tuple(keypoints[limb[1]])
        cv2.line(img_copy, point0, point1, limb_color, 2)
    return img_copy