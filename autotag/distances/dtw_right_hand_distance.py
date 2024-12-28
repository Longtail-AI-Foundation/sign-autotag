import numpy as np
from ..utils.coco_ranges import *
from ..utils.utils import *
from dtw import dtw
from copy import deepcopy

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

def dtw_distance_fn (x, y) : 
    """
    check average l2 difference between visible points
    """
    x = x.reshape(-1, 2)
    y = y.reshape(-1, 2)

    mask = (x > -1) & (y > -1) # select visible points
    mask = mask.all(axis=1) 
    
    return np.linalg.norm(x[mask] - y[mask], axis=1).mean()

def dtw_right_hand_distance(source_kps_and_scores, target_kps_and_scores, window_len, window_stride=1, **kwargs):
    source_kps, source_scores = source_kps_and_scores
    target_kps, target_scores = target_kps_and_scores

    normalized_window_source = normalize_in_bounds(dict(kp=source_kps, kp_sc=source_scores))

    selected_kps = coco_wholebody_right_hand_range 
    if kwargs.get('use_arm', True) : 
        selected_kps = selected_kps + coco_wholebody_right_arm_range 
    n_pts = len(selected_kps)
    right_hand_source = normalized_window_source[:, selected_kps, :]

    distances = []
    n_frames = len(target_kps)

    for i in range(0, n_frames - window_len + 1, window_stride):
        window_body_frames = target_kps[i : i + window_len]
        window_scores      = target_scores[i : i + window_len]

        normalized_window_target = normalize_in_bounds(dict(kp=window_body_frames, kp_sc=window_scores))
        right_hand_target = normalized_window_target[:, selected_kps, :]

        x = right_hand_source.reshape(-1, n_pts * 2)
        y = right_hand_target.reshape(-1, n_pts * 2)

        d, *_ = dtw(x, y, dist=dtw_distance_fn)

        distances.append(d)

    return np.array(distances)

