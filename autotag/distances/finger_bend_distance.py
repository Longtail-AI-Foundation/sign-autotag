""" 
This distance is between features that define which
which fingers are bent and which ones are straight
""" 
import numpy as np
import matplotlib.pyplot as plt
from ..utils.coco_ranges import *
from ..utils.utils import *
from copy import deepcopy

DEBUG = False

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

def bent_finger_histogram (window_frames) : 
    bins = np.zeros(len(right_hand_fingers))
    for t in range(len(window_frames)) : 
        kps = window_frames[t]
        for bin_idx, finger in enumerate(right_hand_fingers) :
            finger_kps = kps[finger, :]
            # a, b, c = finger_kps[0], finger_kps[1], finger_kps[-1]
            # angle = compute_angle(a, b, c) 
            angle = compute_angle_2(*finger_kps)
            if angle > 90.0 :
                bins[bin_idx] += 1
    return bins / len(window_frames)

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

    return np.array(distances)

