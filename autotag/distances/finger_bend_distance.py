""" 
This distance is between features that define which
which fingers are bent and which ones are straight
""" 
import numpy as np
import matplotlib.pyplot as plt
from ..utils.coco_ranges import *
from ..utils.utils import *
from copy import deepcopy
from bisect import bisect_left

right_hand_fingers = [
    list(range(2,5)),   # Thumb
    list(range(5,9)),   # Index Finger
    list(range(9,13)),  # Middle Finger
    list(range(13,17)), # Ring Finger
    list(range(17,21))  # Little Finger
]

DEBUG = False

def normalized(vec):
    norms = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / (norms + 1e-7)

def compute_angle(a, b, c):
    v1 = normalized(b - a)
    v2 = normalized(c - b)
    dotp = np.dot(v1, v2)
    dotp = np.clip(dotp, -1.0, 1.0)
    return np.degrees(np.arccos(dotp))

def compute_angle_2(a, b, c, d):
    v1 = normalized(b - a)
    v2 = normalized(d - c)
    dotp = np.dot(v1, v2)
    dotp = np.clip(dotp, -1.0, 1.0)
    return np.degrees(np.arccos(dotp))

def count_bent_fingers(kps):
    cnt = 0
    for finger in right_hand_fingers:
        finger_kps = kps[finger, :]
        if DEBUG:
            plt.plot(finger_kps[:, 0], finger_kps[:, 1], c='b', alpha=0.5)
        a, b, c = finger_kps[0], finger_kps[1], finger_kps[-1]
        angle = compute_angle(a, b, c)
        if angle > 60.0:
            cnt += 1
    if DEBUG:
        plt.scatter(
            kps[coco_wholebody_right_hand_range, 0],
            kps[coco_wholebody_right_hand_range, 1],
            c='g',
            alpha=0.1
        )
        plt.show()
    return cnt

def bent_finger_histogram(window_frames, **kwargs):
    """
    Computes a histogram of finger bending angles categorized into bins

    Parameters:
        window_frames (numpy array): Shape (num_frames, num_keypoints, 2) - hand keypoints over time.

    Returns:
        np.ndarray: A (num_fingers x num_bins) matrix representing the normalized histogram of angles.
    """
    # Select keypoints for the right hand
    selected_kps_right = coco_wholebody_right_hand_range

    # Extract right-hand keypoints
    right_hand_source = window_frames[:, selected_kps_right, :]

    num_fingers = len(right_hand_fingers)
    angles = list(np.linspace(0, 180, kwargs.get('bins', 4) + 1))
    bins = np.zeros((num_fingers, len(angles) - 1))  # 4 bins per finger

    for t in range(len(window_frames)):
        kps = right_hand_source[t]
        # We'll store the angles for each finger to plot if DEBUG is True
        finger_angles = []

        for finger_idx, finger in enumerate(right_hand_fingers):
            if finger_idx == 0:
                # Thumb
                finger_kps = kps[finger, :]
                angle = compute_angle(finger_kps[0], finger_kps[1], finger_kps[2])
            else:
                # Other fingers
                finger_kps = kps[finger, :]
                angle = compute_angle_2(
                    finger_kps[0], finger_kps[1], finger_kps[2], finger_kps[3]
                )

            assert 0 <= angle <= 180, f'(bent_finger_histogram): Out of range angle detected, angle = {angle}'
            finger_angles.append(angle)

            bin_idx = min(max(0, bisect_left(angles, angle) - 1), len(angles) - 1)
            bins[finger_idx, bin_idx] += 1

        # DEBUG plotting
        if DEBUG:
            plt.figure()
            # Plot the whole body keypoints from window_frames[t], translucent
            plt.scatter(window_frames[t, :, 0], window_frames[t, :, 1], c='gray', alpha=0.3)

            colors = ['r', 'g', 'b', 'y', 'c']
            for finger_idx, finger in enumerate(right_hand_fingers):
                finger_kps = kps[finger, :]
                plt.plot(finger_kps[:, 0], finger_kps[:, 1], color=colors[finger_idx])
                # Place the angle text near the last keypoint
                x_text, y_text = finger_kps[-1, 0], finger_kps[-1, 1]
                plt.text(x_text, y_text, f"{finger_angles[finger_idx]:.2f}", color=colors[finger_idx])

            plt.title(f'Frame {t}')
            plt.show()

    return bins / len(window_frames)

def chi_squared_distance(hist_source, hist_target, epsilon=1e-10):
    """
    Computes the Chi-Squared Distance between two histograms.

    Parameters:
        hist_source (np.ndarray): Source histogram (num_fingers x 4).
        hist_target (np.ndarray): Target histogram (num_fingers x 4).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Chi-squared distance for each finger.
    """
    return 0.5 * np.sum(((hist_source - hist_target) ** 2) / (hist_source + hist_target + epsilon), axis=1)

def finger_bend_distance(source_kps_and_scores, target_kps_and_scores, window_len, window_stride=1, **kwargs):
    source_kps, source_scores = source_kps_and_scores
    target_kps, target_scores = target_kps_and_scores

    source_hist = bent_finger_histogram(source_kps, **kwargs)

    distances = []
    n_frames = len(target_kps)

    for i in range(0, n_frames - window_len + 1, window_stride):
        window_body_frames = target_kps[i : i + window_len]
        target_hist = bent_finger_histogram(window_body_frames, **kwargs)
        d = chi_squared_distance(source_hist, target_hist).sum()
        distances.append(d)

    return np.array(distances)

