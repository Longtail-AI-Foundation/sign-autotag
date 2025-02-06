""" 
This distance is between features that define which
which fingers are bent and which ones are straight
""" 
""" 
This distance is between features that define which
which fingers are bent and which ones are straight
""" 
import numpy as np
import matplotlib.pyplot as plt
from ..utils.coco_ranges import *
from ..utils.utils import *
from copy import deepcopy

right_hand_fingers = [
    list(range(2,5)),   # Thumb
    list(range(5,9)),   # Index Finger
    list(range(9,13)),  # Middle Finger
    list(range(13,17)),    # Ring Finger
    list(range(17,21))   # Little Finger
]

DEBUG = False

def normalized (vec):
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

def count_bent_fingers (kps) : 
    cnt = 0
    for finger in right_hand_fingers:  
        finger_kps = kps[finger, :]
        if DEBUG: 
            plt.plot(finger_kps[:, 0], finger_kps[:, 1], c='b', alpha=0.5)
        a, b, c = finger_kps[0], finger_kps[1], finger_kps[-1]
        angle = compute_angle(a, b, c) 
        if angle > 60.0 :
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
    Computes a histogram of finger bending angles categorized into four bins: 
    [0-45°, 45-90°, 90-135°, 135-180°] for each finger.

    Parameters:
        window_frames (numpy array): Shape (num_frames, num_keypoints, 2 or 3) - hand keypoints over time.
        kwargs: Optional parameters such as 'use_arm' (bool).

    Returns:
        np.ndarray: A (num_fingers x 4) matrix representing the normalized histogram of angles.
    """
    # Select keypoints for the right hand
    selected_kps_right = coco_wholebody_right_hand_range
    if kwargs.get('use_arm', True):
        selected_kps_right += coco_wholebody_right_arm_range

    

    # Extract right-hand keypoints
    right_hand_source = window_frames[:, selected_kps_right, :]

    num_fingers = len(right_hand_fingers)
    bins = np.zeros((num_fingers, 4))  # 4 bins per finger

    for t in range(len(window_frames)):
        kps = right_hand_source[t]  # Extract keypoints for current frame
        for finger_idx, finger in enumerate(right_hand_fingers):
            if(finger_idx==0):

                finger_kps = kps[finger, :]
                angle = compute_angle(finger_kps[0],finger_kps[1],finger_kps[2])  # Compute the finger bend angle
            elif (finger_idx!=0):
                finger_kps = kps[finger, :]
                angle = compute_angle_2(finger_kps[0],finger_kps[1],finger_kps[2],finger_kps[3])
            # Assign angle to appropriate bin
            if 0 <= angle < 45:
                bins[finger_idx, 0] += 1
            elif 45 <= angle < 90:
                bins[finger_idx, 1] += 1
            elif 90 <= angle < 135:
                bins[finger_idx, 2] += 1
            elif 135 <= angle <= 180:
                bins[finger_idx, 3] += 1

    return bins / len(window_frames)  # Normalize histogram

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

    source_hist = bent_finger_histogram(source_kps) 

    distances = []
    n_frames = len(target_kps)

    for i in range(0, n_frames - window_len + 1, window_stride):
        window_body_frames = target_kps[i : i + window_len]
        target_hist = bent_finger_histogram(window_body_frames)
        d = chi_squared_distance(source_hist, target_hist)
        distances.append(d)
    distance=np.sum(distances,axis=1)
    return np.array(distances)

