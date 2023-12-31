import numpy as np
from manotorch.utils.anchorutils import get_region_palm_mask


def pairwise_dist(x, y):
    n_x, n_y = x.shape[0], y.shape[0]
    zz = np.matmul(x, y.transpose(1, 0))
    rx = np.repeat(np.expand_dims(np.sum(x**2, axis=1), axis=1), n_y, axis=1)
    ry = np.repeat(np.expand_dims(np.sum(y**2, axis=1), axis=0), n_x, axis=0)
    P = rx + ry - 2 * zz
    return P


def region_disjointness_metric(hand_verts, obj_verts, hand_region_assignment):
    """disjointedness calculate the average distance of hand vertices in the five-finger region 
    to the nearest object surface

    Args:
        hand_verts (np.ndarray): hand vertices, (778, 3)
        obj_verts (np.ndarray): object vertices
        hand_region_assignment (np.ndarray): indicates which region the hand vertices belong to, (778, )

    Returns:
        res (np.array): min distance of hand vertices in each region to the nearest object surface, (17, )
        tip_only (float): sum of min distance of hand vertices in tip region to the nearest object surface. 

    """
    tip_regions = [2, 5, 8, 11, 14]  # tip region constant
    tip_only_weight = np.zeros(17, dtype=np.float64)
    tip_only_weight[tip_regions] = 1.0
    # first get all possible regions
    all_regions = list(range(17))
    res_list = []
    # iterate over regions
    for region_id in all_regions:
        # get boolean mask
        region_mask = get_region_palm_mask(region_id, None, hand_region_assignment, None)
        # select hand points
        hand_verts_region = hand_verts[region_mask]
        # cross distance
        dist_mat = np.abs(pairwise_dist(hand_verts_region, obj_verts))  # make sure positive
        # compute min hand -> object
        hand_to_object_dist = np.min(dist_mat, axis=1)
        # sqrt
        hand_to_object_dist = np.sqrt(hand_to_object_dist)
        # compute min value over region
        min_value = np.min(hand_to_object_dist)
        # get into
        res_list.append(min_value)
    res = np.array(res_list)
    tip_only = np.sum(res * tip_only_weight)
    return res, tip_only
