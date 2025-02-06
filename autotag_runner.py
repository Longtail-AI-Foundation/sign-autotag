import argparse
import pdb
import os
import numpy as np
from collections import defaultdict
from matplotlib.animation import FuncAnimation 
from autotag.utils.utils import *
from autotag.utils.vis_utils import *
from autotag.distances.body_range_distance import body_range_distance
from autotag.distances.hands_used_distance import hands_used_distance
from autotag.distances.dtw_right_hand_distance import dtw_right_hand_distance
from autotag.distances.finger_bend_distance import finger_bend_distance

def _main(sourcefile, source_dir, target_dir) :
    source_data = read_json(sourcefile)

    source_video = os.path.basename(source_data["source"]["source_video"])
    basename = os.path.splitext(source_video)[0]    
    source_video = os.path.join(source_dir, source_video)
    
    source_pkl = os.path.join(source_dir, f"{basename}.pkl")

    source_start_ts = source_data["source"]["signVideos"][0]["signStartTS"]
    source_end_ts = source_data["source"]["signVideos"][0]["signEndTS"]
    source_fps = source_data["source"]["fps"]

    source_pkl_data = open_keypoints(source_pkl)

    target_video = os.path.basename(source_data["target"]["signVideos"][0]["targetFilePath"])  
    basename = os.path.splitext(target_video)[0]    
    target_video = os.path.join(target_dir, target_video) # make FQN
    target_pkl = os.path.join(target_dir, f"{basename}.pkl")
    target_start_ts = source_data["target"]["signVideos"][0]["signStartTS"]
    target_end_ts = source_data["target"]["signVideos"][0]["signEndTS"]
    target_fps = source_data["target"]["signVideos"][0]["fps"]
    
    target_pkl_data = open_keypoints(target_pkl)

    kp_source_all = extract_keypoints_sequence(source_pkl_data)
    kp_source_segment, source_start_frame, source_end_frame = get_keypoints_for_timerange(kp_source_all, source_fps, source_start_ts, source_end_ts)

    kp_target_all = extract_keypoints_sequence(target_pkl_data)
    kp_target_segment, target_start_frame, target_end_frame = get_keypoints_for_timerange(kp_target_all, target_fps, target_start_ts, target_end_ts)

    true_window_len = target_end_frame - target_start_frame + 1
    print('True target window length = ', true_window_len)

    range_distances = body_range_distance(kp_source_segment, kp_target_all, true_window_len)
    hand_distances = hands_used_distance(kp_source_segment, kp_target_all, true_window_len)
    dtw_distances = dtw_right_hand_distance(kp_source_segment, kp_target_all, true_window_len, use_arm=True)

    finger_bend_distances = finger_bend_distance(kp_source_segment, kp_target_all, true_window_len) 

    distances = finger_bend_distances
    plot_distance(distances, true_start=target_start_frame, true_end=target_end_frame)
    annotate_and_show_video(target_video, distances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Something to do with keypoint analysis, okay!')
    parser.add_argument('--source', type=str, required=True, help='path to the source json file')
    parser.add_argument('--source_video_dir', type=str, required=True, help='source video dir')
    parser.add_argument('--target_video_dir', type=str, required=True, help='target video dir')

    args = parser.parse_args()
    _main(args.source, args.source_video_dir, args.target_video_dir)
