import numpy as np 
import matplotlib.pyplot as plt
from ..utils.coco_ranges import *
from ..utils.utils import *

DEBUG = False

def extract_body_region_ranges(keypoints):
    """
    Defines 4 contiguous bins for body regions (above head, head, chest, mid/body).
    Assumes y is larger going down, so "above head" corresponds to smaller y-values.
    """
    body_mid_range = (12, 13)
    chest_range    = (6, 7)
    head_range     = (2, 3)

    body_mid = np.array(get_kps_for_range(keypoints, body_mid_range))
    chest    = np.array(get_kps_for_range(keypoints, chest_range))
    head     = np.array(get_kps_for_range(keypoints, head_range))

    b_mid   = float(body_mid[..., 1].mean())   # average y for mid
    b_chest = float(chest[..., 1].mean())      # average y for chest
    b_head  = float(head[..., 1].mean())       # average y for head

    levels = b_head, b_chest, b_mid

    # We define 4 bins, contiguous:
    #   1) (-∞, levels[0])          -> "above head"
    #   2) [levels[0], levels[1])   -> head region
    #   3) [levels[1], levels[2])   -> chest region
    #   4) [levels[2], +∞)          -> mid (down to legs, etc.)

    body_points = []
    body_points.append((0.0, levels[0]))   # above head
    body_points.append((levels[0], levels[1]))  # head
    body_points.append((levels[1], levels[2]))  # chest
    body_points.append((levels[2], 1e7))    # mid/lower

    return body_points

def calculate_histogram_for_sign(right_hand_keypoints, body_points):
    histogram = [0, 0, 0, 0]
    for frame_hand_keypoints in right_hand_keypoints:
        for keypoint in frame_hand_keypoints:
            y = keypoint[1]
            if body_points[0][0] <= y <= body_points[0][1]:
                histogram[0] += 1
            elif body_points[1][0] <= y <= body_points[1][1]:
                histogram[1] += 1
            elif body_points[2][0] <= y <= body_points[2][1]:
                histogram[2] += 1
            elif y >= body_points[3][0]:
                histogram[3] += 1
    total = sum(histogram)
    if total > 0:
        histogram = [h / total for h in histogram]
    return histogram

def histogram_distance(source_hist, target_hist):
    source_hist = np.array(source_hist, dtype=float)
    target_hist = np.array(target_hist, dtype=float)
    if source_hist.sum() == 0 or target_hist.sum() == 0:
        return 10.0
    source_hist /= source_hist.sum()
    target_hist /= target_hist.sum()
    diff = np.abs(source_hist - target_hist)
    if np.any(diff > 0.5):
        return 10.0
    return diff.sum()

def body_range_distance(source_kps_and_scores, target_kps_and_scores, window_len, window_stride=1, **kwargs):

    source_kps, _ = source_kps_and_scores
    target_kps, _ = target_kps_and_scores

    body_points_source = extract_body_region_ranges(source_kps)
    right_hand_source  = get_kps_for_range(source_kps, coco_wholebody_right_hand_range, one_indexed=False)
    source_hist        = calculate_histogram_for_sign(right_hand_source, body_points_source)

    right_hand_target = get_kps_for_range(target_kps, coco_wholebody_right_hand_range, one_indexed=False)
    distances = []
    n_frames = len(right_hand_target)

    for i in range(0, n_frames - window_len + 1, window_stride):
        window_frames      = right_hand_target[i : i + window_len]
        window_body_frames = target_kps[i : i + window_len]
        body_points_target = extract_body_region_ranges(window_body_frames)
        target_hist        = calculate_histogram_for_sign(window_frames, body_points_target)
        dist               = histogram_distance(source_hist, target_hist)
        distances.append(dist)

        if DEBUG :
            fig, ax = plt.subplots()
            # Plot the first frame's body keypoints
            if len(window_body_frames) > 0 and len(window_body_frames[0]) > 0:
                x_vals = [kp[0] for kp in window_body_frames[0]]
                y_vals = [kp[1] for kp in window_body_frames[0]]
                ax.scatter(x_vals, y_vals, c='blue', alpha=0.4)

                ax.scatter(right_hand_target[i : i + window_len][..., 0], right_hand_target[i : i + window_len][..., 1], c='green', alpha=1.0)

                # Plot only the low line for each bin
                for bin_index, (low, high) in enumerate(body_points_target):
                    ax.axhline(y=low, color='red', linestyle='--', linewidth=1.0)
                    # Write bin value near the low line
                    ax.text(
                        min(x_vals) - 10,
                        low,
                        f"{target_hist[bin_index]:.2f}",
                        color='red',
                        fontsize=12,
                        backgroundcolor='white',
                        ha='right'
                    )

                # Distance text in the top center
                ax.text(
                    0.5,
                    0.95,
                    f"Dist: {dist:.2f}",
                    color='magenta',
                    fontsize=12,
                    fontweight='bold',
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    backgroundcolor='white'
                )

                # Adjust view if possible
                if y_vals:
                    y_min, y_max = min(y_vals), max(y_vals)
                    span = y_max - y_min if (y_max - y_min) > 0 else 10
                    ax.set_ylim([0, y_max + 0.2 * span])

            plt.show()

    return np.array(distances)

