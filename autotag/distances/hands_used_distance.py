import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from enum import Enum
from ..utils.coco_ranges import *
from ..utils.utils import *

DEBUG = False

class HandUsage(Enum):
    SINGLE_HAND = 0
    BOTH_HAND   = 1

def meets_threshold(array, threshold=0.8):
    """
    Check if the proportion of True values in the array meets or exceeds the threshold.

    Parameters:
    array (np.ndarray): A boolean array.
    threshold (float): Proportion threshold (default is 0.8).

    Returns:
    bool: True if proportion of True values >= threshold, else False.
    """
    return (array.sum() / array.size) >= threshold

def normalize_in_bounds(data):
    data = deepcopy(data)
    kpss = np.stack(data['kp'])
    kpsc = np.stack(data['kp_sc'])

    visible_mask = (kpsc > 0.5) & (~np.isnan(kpss).any(-1))
    vis_kps = kpss[visible_mask]
    if len(vis_kps) == 0:
        return np.full_like(kpss, -1.0)

    x, X = vis_kps[..., 0].min(), vis_kps[..., 0].max()
    y, Y = vis_kps[..., 1].min(), vis_kps[..., 1].max()

    # Center
    kpss[..., 0] -= x + (X - x) / 2
    kpss[..., 1] -= y + (Y - y) / 2

    # Scale
    S = max(X - x, Y - y)
    kpss /= (S / 2)

    # Flip Y
    kpss[..., 1] = -kpss[..., 1]

    # Mark invisible
    kpss[~visible_mask] = -2.0
    return kpss

def compute_angle(shoulder, elbow, wrist):
    v1 = normalized(shoulder - elbow)
    v2 = normalized(wrist   - elbow)
    dotp = np.dot(v1, v2)
    dotp = np.clip(dotp, -1.0, 1.0)
    return np.degrees(np.arccos(dotp))

def compute_movement(kpss, arm_range):
    diffs = []
    for t in range(1, len(kpss)):
        prev_arm = kpss[t - 1, arm_range]
        if DEBUG :
            plt.scatter(prev_arm[:, 0], prev_arm[:, 1], c='b', alpha=t/len(kpss))
        curr_arm = kpss[t, arm_range]
        if (prev_arm < -1.0).any() or (curr_arm < -1.0).any():
            diffs.append(0.0)
        else:
            diffs.append(np.linalg.norm(curr_arm - prev_arm, axis=1).sum())
    if DEBUG :
        plt.show()
    return np.sum(diffs) if diffs else 0.0

def hand_movement_heuristic(data, angle_threshold=30, move_factor=0.60):
    kpss = normalize_in_bounds(data)
    kpsc = data['kp_sc']
    T = kpss.shape[0]

    # Visibility check
    left_hand_visible  = meets_threshold(kpsc[:, coco_wholebody_left_hand_range] > 0.5, threshold=0.8)
    right_hand_visible = meets_threshold(kpsc[:, coco_wholebody_right_hand_range] > 0.5, threshold=0.8)
    if not left_hand_visible and not right_hand_visible:
        return HandUsage.SINGLE_HAND
    if not left_hand_visible and right_hand_visible:
        return HandUsage.SINGLE_HAND
    if left_hand_visible and not right_hand_visible:
        return HandUsage.SINGLE_HAND

    # Compute angles
    left_angles, right_angles = [], []
    for t in range(T):
        l_arm = kpss[t, coco_wholebody_left_arm_range]
        r_arm = kpss[t, coco_wholebody_right_arm_range]

        if (l_arm < -1.0).any():
            left_angles.append(0.0)
        else:
            left_angles.append(compute_angle(l_arm[0], l_arm[1], l_arm[2]))

        if (r_arm < -1.0).any():
            right_angles.append(0.0)
        else:
            right_angles.append(compute_angle(r_arm[0], r_arm[1], r_arm[2]))

    left_used  = any(a < angle_threshold for a in left_angles)
    right_used = any(a < angle_threshold for a in right_angles)

    if left_used and right_used:
        return HandUsage.BOTH_HAND

    left_used = (max(left_angles) - min(left_angles)) > 30
    right_used = (max(right_angles) - min(right_angles)) > 30

    if left_used and right_used :
        return HandUsage.BOTH_HAND

    # Compute movement
    left_move  = compute_movement(kpss, coco_wholebody_left_hand_range)
    right_move = compute_movement(kpss, coco_wholebody_right_hand_range)

    if left_move > (move_factor * right_move) : 
        return HandUsage.BOTH_HAND

    return HandUsage.SINGLE_HAND

def hands_used_distance(source_kps_and_scores, target_kps_and_scores, window_len, window_stride=1, **kwargs):
    source_kps, source_scores = source_kps_and_scores
    target_kps, target_scores = target_kps_and_scores

    source_label = hand_movement_heuristic(dict(kp=source_kps, kp_sc=source_scores))

    if DEBUG : 
        print(source_label)

    distances = []
    n_frames = len(target_kps)
    for i in range(0, n_frames - window_len + 1, window_stride):
        window_body_frames = target_kps[i : i + window_len]
        window_scores      = target_scores[i : i + window_len]
        target_window_label = hand_movement_heuristic(dict(kp=window_body_frames, kp_sc=window_scores))

        if DEBUG : 
            print(target_window_label)

        dist = 10.0 if (target_window_label != source_label) else 0.0
        distances.append(dist)

    return np.array(distances)

