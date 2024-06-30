import numpy as np
import torch
import math


# функции добавляют и убирают 1 к массиву для расчета аффинного преобразования
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:, :-1]


def affine_transform(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Affine transform points2 to points1

    Args:
        points1 (np.ndarray): ethalon points
        points2 (np.ndarray): target points

    Returns:
        np.ndarray: output points
    """
    Y = pad(points1)
    X = pad(points2)
    A, _, _, _ = np.linalg.lstsq(X, Y)
    A[np.abs(A) < 1e-10] = 0
    transform = lambda x: unpad(np.dot(pad(x), A))
    points2_1 = transform(points2)
    return points2_1


def get_similarity(points1: np.ndarray, points2: np.ndarray) -> float:
    """Get cosine similarity of two poses

    Args:
        points1 (np.ndarray): point of first pose
        points2 (np.ndarray): points of second pose

    Returns:
        float: cosine similarity
    """
    points2 = affine_transform(points1, points2)
    sim = torch.nn.functional.cosine_similarity(torch.Tensor(points1), torch.Tensor(points2))
    return sim.mean().item()


def weight_distance(pose1: np.ndarray, pose2: np.ndarray, conf1: float) -> float:
    """Get weighted distance of two poses

    Args:
        pose1 (np.ndarray): point of first pose
        pose2 (np.ndarray): points of second pose
        conf1 (float): confidence of predicted points of first pose

    Returns:
        float: weighted distance
    """
    pose2 = affine_transform(pose1, pose2)
    sum1 = 1 / np.sum(conf1)
    sum2 = 0
    for i in range(len(pose1)):
        sum2 += conf1[i] * abs(math.hypot(pose1[i][0] - pose2[i][0], pose1[i][1] - pose2[i][1]))
    weighted_dist = sum1 * sum2

    return weighted_dist