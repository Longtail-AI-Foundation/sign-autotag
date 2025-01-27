import pickle
import argparse
import pdb
import json
import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import correlation
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cosine
# from fastdtw import fastdtw
# from dtw import accelerated_dtw
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import cv2
from autotag.utils.utils import *
from autotag.utils.vis_utils import *
from autotag.distances.body_range_distance import body_range_distance
from autotag.distances.hands_used_distance import hands_used_distance
from autotag.distances.dtw_right_hand_distance import dtw_right_hand_distance
from autotag.distances.fingertip_movement_distance import fingertip_movement_distance

OUTPUT_PATH = 'frames'
THRESHOLD = 4.5

def find_minimum_dtw(dtw_distances):
    """Find the index of the minimum DTW distance."""
    min_index = np.argmin(dtw_distances)
    min_distance = dtw_distances[min_index]
    return min_index, min_distance


def sliding_window_combined_distance(source_kp, target_kp, right_hand_source, right_hand_target, window_size):
    """
    Combines histogram distance and DTW distance for sliding windows in the target video.
    
    Parameters:
        source_kp (list): Keypoints from the source video.
        target_kp (list): Keypoints from the target video.
        right_hand_source (list): Right hand keypoints from the source video.
        right_hand_target (list): Right hand keypoints from the target video.
        window_size (int): Number of frames in the source video (used as the window size).
    
    Returns:
        list: Combined distance values (histogram + DTW) for each window in the target video.
    """
    # Calculate the source histogram
    body_points_target = extract_body_region_ranges(target_kp)
    body_points_source = extract_body_region_ranges(source_kp)
    source_hist = calculate_histogram_for_sign(right_hand_source, body_points_source)

    # Normalize the source keypoints for DTW calculation
    source_normalized = normalise_keypoints(right_hand_source)
    source_reshaped = source_normalized.reshape(window_size, -1)  # Reshape for DTW computation

    combined_distances = []

    # Iterate through the target video using the sliding window
    for i in range(len(right_hand_target) - window_size):
        # Get the frames for the current window
        new_window=window_size//2
        
        window_frames = right_hand_target[i:i + new_window]

        # Calculate the histogram for the current window
        target_hist = calculate_histogram_for_sign(window_frames, body_points_target)
        
        # Compute the histogram distance
        hist_distance = histogram_distance(source_hist, target_hist)

        # Normalize the window keypoints for DTW calculation
        window_normalized = normalise_keypoints(window_frames)
        window_reshaped = window_normalized.reshape(new_window, -1)

        # Compute DTW distance
        dtw_distance, _, _, _ = accelerated_dtw(source_reshaped, window_reshaped, dist='cosine')

        # Combine the distances
        combined_distance = hist_distance + dtw_distance
        combined_distances.append(combined_distance)

    return combined_distances


def calculate_movement_signature(keypoints):
    # Updated bin ranges in degrees (adjusted for monotonicity)
    bin_edges_degrees = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    bin_edges = np.radians(bin_edges_degrees)  # Convert degrees to radians
    histogram = np.zeros(len(bin_edges) - 1)  # 8 bins for the histogram

    # Calculate differences between consecutive frames
    frame_differences = np.diff(keypoints, axis=0)  # Shape: [num_frames-1, 21, 2]

    for frame_diff in frame_differences:
        # Compute the average movement vector for this frame (average across 21 points)
        average_movement = np.mean(frame_diff, axis=0)  # Shape: [2] (vx, vy)
        movement_magnitude = np.linalg.norm(average_movement)  # Scalar

        # Skip frames with negligible movement
        if np.isclose(movement_magnitude, 0):
            continue

        # Compute the movement angle in radians
        angle = np.arctan2(average_movement[1], average_movement[0])  # Radians (-π to π)

        # Convert angle to range [0, 2π] (equivalent to 0° to 360°)
        if angle < 0:
            angle += 2 * np.pi

        # Bin the angle into one of the 8 bins
        for j in range(len(bin_edges) - 1):
            if bin_edges[j] <= angle < bin_edges[j + 1]:
                histogram[j] += movement_magnitude  # Add weighted magnitude to the bin
                break

        # Normalize the histogram to ensure it sums to 1
    if np.sum(histogram) > 0:  # Avoid division by zero
        histogram /= np.sum(histogram)

    return histogram

def calculate_movement_signature_fingertips(keypoints, bottom_hand, bottom_middle_finger):
    # Updated bin ranges in degrees (adjusted for monotonicity)
    bin_edges_degrees = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    bin_edges = np.radians(bin_edges_degrees)  # Convert degrees to radians
    histogram = np.zeros(len(bin_edges) - 1)  # 8 bins for the histogram

    # Calculate differences between consecutive frames
    # print(len(keypoints))
    # print(len(bottom_hand))
    # print(len(bottom_middle_finger))
    frame_differences = np.diff(keypoints, axis=0)  # Shape: [num_frames-1, 21, 2]
    # frame_difference_hand=np.diff(bottom_hand,axis=0)
    # frame_difference_finger=np.diff(bottom_middle_finger,axis=0)
    palm_position = (np.array(bottom_hand) + np.array(bottom_middle_finger)) / 2
    palm_frame=np.diff(palm_position,axis=0)
    # print(len(frame_differences))
    # print(len(frame_difference_hand))
    # print(len(frame_difference_finger))

    for frame_idx, frame_diff in enumerate(frame_differences):
        # Current and previous frame points
        current_frame = frame_differences[frame_idx]  # Shape: [21, 2]
        
        # Palm position: average of bottom-middle-finger and bottom-of-hand points
        # bottom_hand = current_frame[bottom_hand_index]  # Bottom of hand
        # bottom_middle_finger = current_frame[bottom_middle_finger_index]  # Bottom of middle finger
        # bottom_hand_frame=bottom_hand[frame_idx+1]
        # bottom_middle_finger_frame=bottom_middle_finger[frame_idx+1]
        
        # palm_position = (np.array(bottom_hand_frame) + np.array(bottom_middle_finger_frame)) / 2 # Average palm position
        
        # Subtract the palm position from every point in the current frame
        # print(len(current_frame))
        # print(palm_position.shape)
        palm_curr_frame=palm_frame[frame_idx]
        relative_positions = current_frame - palm_curr_frame  # Shape: [21, 2]
        # print(relative_positions)
        
        # Compute the average movement vector relative to the palm
        average_movement = np.mean(relative_positions, axis=0)  # Shape: [2] (vx, vy)
        movement_magnitude = np.linalg.norm(average_movement)  # Scalar

        # Skip frames with negligible movement
        if np.isclose(movement_magnitude, 0):
            continue

        # Compute the movement angle in radians
        angle = np.arctan2(average_movement[1], average_movement[0])  # Radians (-π to π)

        # Convert angle to range [0, 2π] (0° to 360°)
        if angle < 0:
            angle += 2 * np.pi

        # Bin the angle into one of the 8 bins
        for j in range(len(bin_edges) - 1):
            if bin_edges[j] <= angle < bin_edges[j + 1]:
                histogram[j] += movement_magnitude  # Add weighted magnitude to the bin
                break

    # Normalize the histogram to ensure it sums to 1
    hist_min = np.min(histogram)
    hist_max = np.max(histogram)

    if np.sum(histogram) > 0:  # Avoid division by zero
        histogram /= np.sum(histogram)

    return histogram

def calculate_movement_signature_allfingertips(finger1_keypoints, finger2_keypoints, finger3_keypoints, finger4_keypoints, finger5_keypoints, bottom_hand, bottom_middle_finger):
    """
    Calculate movement signature histograms for 5 fingertips and combine them into a final histogram.
    
    Parameters:
        fingerX_keypoints: Numpy arrays of shape [num_frames, 2], containing (x, y) coordinates for each fingertip.
        bottom_hand: Numpy array of shape [num_frames, 2], containing the (x, y) coordinates for the bottom of the hand.
        bottom_middle_finger: Numpy array of shape [num_frames, 2], containing the (x, y) coordinates for the bottom of the middle finger.

    Returns:
        final_histogram: Combined histogram for all fingertips (normalized to sum to 1).
        individual_histograms: List of histograms for each fingertip.
        contributions: Contributions of each fingertip to the final histogram.
    """
    # Updated bin ranges in degrees (adjusted for monotonicity)
    bin_edges_degrees = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    bin_edges = np.radians(bin_edges_degrees)  # Convert degrees to radians
    num_bins = len(bin_edges) - 1

    # Initialize variables
    finger_keypoints = [finger1_keypoints, finger2_keypoints, finger3_keypoints, finger4_keypoints, finger5_keypoints]
    individual_histograms = []
    contributions = []
    total_movement_magnitude = 0

    # Calculate the palm position for all frames
    palm_position = (np.array(bottom_hand) + np.array(bottom_middle_finger)) / 2
    palm_frame_differences = np.diff(palm_position, axis=0)  # Shape: [num_frames-1, 2]

    # Process each fingertip
    for finger_idx, finger_keypoint in enumerate(finger_keypoints):
        histogram = np.zeros(num_bins)  # Initialize histogram for the current fingertip
        frame_differences = np.diff(finger_keypoint, axis=0)  # Shape: [num_frames-1, 2]

        fingertip_movement_magnitude = 0  # Total movement magnitude for the current fingertip

        for frame_idx, frame_diff in enumerate(frame_differences):
            # Compute the relative position of the fingertip with respect to the palm
            palm_curr_frame = palm_frame_differences[frame_idx]
            # print(frame_diff.shape)
            # print(palm_curr_frame.shape)
            # current_frame_diff=frame_diff[frame_idx]
            
            relative_position = frame_diff - palm_curr_frame  # Shape: [2] (vx, vy)
            # print(relative_position.shape)
            # Compute movement magnitude and angle
            average_movement = np.mean(relative_position, axis=0)  # Shape: [2] (vx, vy)
            movement_magnitude = np.linalg.norm(average_movement)
            # movement_magnitude = np.linalg.norm(relative_position)  # Scalar
            if np.isclose(movement_magnitude, 0):  # Skip negligible movement
                continue

            angle = np.arctan2(average_movement[1], average_movement[0])  # Radians (-π to π)
            if angle < 0:
                angle += 2 * np.pi  # Convert to range [0, 2π]

            # Bin the angle
            for j in range(num_bins):
                if bin_edges[j] <= angle < bin_edges[j + 1]:
                    histogram[j] += movement_magnitude  # Add weighted magnitude to the bin
                    break

            # Accumulate movement magnitude
            fingertip_movement_magnitude += movement_magnitude
        
        # print(fingertip_movement_magnitude)
        # Append individual histogram and contribution
        individual_histograms.append(histogram)
        contributions.append(fingertip_movement_magnitude)
        total_movement_magnitude += fingertip_movement_magnitude

    # Normalize contributions to calculate weights for combining histograms
    contributions = np.array(contributions)
    if total_movement_magnitude > 0:
        weights = contributions / total_movement_magnitude
    else:
        weights = np.zeros(len(contributions))  # Handle edge case

    # Combine individual histograms based on weights
    final_histogram = np.zeros(num_bins)
    for i, histogram in enumerate(individual_histograms):
        final_histogram += weights[i] * histogram

    # Normalize the final histogram to sum to 1
    if np.sum(final_histogram) > 0:  # Avoid division by zero
        final_histogram /= np.sum(final_histogram)

    return final_histogram, individual_histograms, contributions

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

    # Assume right handed person.
    left_hand_range = range(91, 112)
    right_hand_range = range(113, 134)

    right_finger_thumb_range=range(117,118)
    right_middlefing_bottom_range=range(122,123)
    right_hand_bottom_range=range(113,114)
    right_indexfinger_range=range(121,122)
    right_middlefing_range=range(125,126)
    right_ringfinger_range=range(129,130)
    right_littlefinger_range=range(133,134)

    kp_source_all = extract_keypoints_sequence(source_pkl_data)
    kp_source_segment, source_start_frame, source_end_frame = get_keypoints_for_timerange(kp_source_all, source_fps, source_start_ts, source_end_ts)

    right_hand_source_kp = get_kps_for_range(kp_source_segment[0], right_hand_range)
    right_handall_source_kp=get_kps_for_range(kp_source_all[0],right_hand_range)
    right_finger_thumb_source_kp=get_kps_for_range(kp_source_segment[0],right_finger_thumb_range)
    right_middlefing_bottom_source_kp=get_kps_for_range(kp_source_segment[0],right_middlefing_bottom_range)
    right_hand_bottom_source_kp=get_kps_for_range(kp_source_segment[0],right_hand_bottom_range)
    right_indexfinger_source_kp=get_kps_for_range(kp_source_segment[0],right_indexfinger_range)
    right_middlefinger_source_kp=get_kps_for_range(kp_source_segment[0],right_middlefing_range)
    right_ringfinger_source_kp=get_kps_for_range(kp_source_segment[0],right_ringfinger_range)
    right_littlefinger_source_kp=get_kps_for_range(kp_source_segment[0],right_littlefinger_range)

    kp_target_all = extract_keypoints_sequence(target_pkl_data)
    kp_target_segment, target_start_frame, target_end_frame = get_keypoints_for_timerange(kp_target_all, target_fps, target_start_ts, target_end_ts)

    # try with 5 also
    range_distances = body_range_distance(kp_source_segment, kp_target_all, right_hand_source_kp.shape[0]//2)
    hand_distances = hands_used_distance(kp_source_segment, kp_target_all, right_hand_source_kp.shape[0]//2)
    dtw_distances = dtw_right_hand_distance(kp_source_segment, kp_target_all, right_hand_source_kp.shape[0]//2, use_arm=True)
    movement_distance=fingertip_movement_distance(kp_source_segment, kp_target_all, right_hand_source_kp.shape[0]//2)

    plot_distance(hand_distances, true_start=target_start_frame, true_end=target_end_frame)

    distances = (range_distances + hand_distances + dtw_distances + movement_distance) / 4
    plot_distance(distances, true_start=target_start_frame, true_end=target_end_frame)
    annotate_and_show_video(target_video, distances)
    exit()

    right_hand_target_kp = get_kps_for_range(kp_target_all, right_hand_range)
    right_hand_targetseg_kp=get_kps_for_range(kp_target_segment,right_hand_range)
    right_finger_thumb_target_kp=get_kps_for_range(kp_target_segment,right_finger_thumb_range)
    right_middlefing_bottom_target_kp=get_kps_for_range(kp_target_segment,right_middlefing_bottom_range)
    right_hand_bottom_target_kp=get_kps_for_range(kp_target_segment,right_hand_bottom_range)
    right_indexfinger_target_kp=get_kps_for_range(kp_target_segment,right_indexfinger_range)
    right_middlefinger_target_kp=get_kps_for_range(kp_target_segment,right_middlefing_range)
    right_ringfinger_target_kp=get_kps_for_range(kp_target_segment,right_ringfinger_range)
    right_littlefinger_target_kp=get_kps_for_range(kp_target_segment,right_littlefinger_range)
    # right_hand_target_kp = get_kps_for_range(kp_target_all, right_hand_range)

    source_frame_range = range(source_start_frame, source_end_frame + 1)
    target_frame_range = range(target_start_frame, target_end_frame + 1)
    
    right_hand_source_np = normalise_keypoints(right_hand_source_kp)
    right_hand_target_np = normalise_keypoints(right_hand_target_kp)
    right_hand_targetseg_np=normalise_keypoints(right_hand_targetseg_kp)

    if source_data["source"]["handedness"] == "both" :
        left_hand_source_kp = get_kps_for_range(extract_keypoints_sequence(kp_source_segment), left_hand_range) #retrieve left hand kp across frames
        left_hand_target_kp = get_kps_for_range(extract_keypoints_sequence(kp_target_all), left_hand_range) #retrieve left hand kp across frames


    


    bin_labels = [
        'Right (0°–45°)', 
        'Up (45°–90°)', 
        'Up (90°–135°)', 
        'Left (135°–180°)', 
        'Left (180°–225°)', 
        'Down (225°–270°)', 
        'Down (270°–315°)', 
        'Right (315°–360°)'
    ]
    # # plot_movement_signatures_both(movement_signature_source, movement_signature_target, bin_labels)

    source_final_histogram, source_individual_histograms, contributions_source=calculate_movement_signature_allfingertips(right_finger_thumb_source_kp, right_indexfinger_source_kp,right_middlefinger_source_kp , right_ringfinger_source_kp, right_littlefinger_source_kp, right_hand_bottom_source_kp,right_middlefing_bottom_source_kp)
    target_final_histogram, target_individual_histograms, contributions = calculate_movement_signature_allfingertips(right_finger_thumb_target_kp, right_indexfinger_target_kp,right_middlefinger_target_kp , right_ringfinger_target_kp, right_littlefinger_target_kp, right_hand_bottom_target_kp,right_middlefing_bottom_target_kp)
    # Plot the histograms
    # plot_fingertip_histograms(final_histogram, individual_histograms, contributions, bin_labels)

    plot_source_target_histograms(
        source_final_histogram=source_final_histogram,
        target_final_histogram=target_final_histogram,
        source_individual_histograms=source_individual_histograms,
        target_individual_histograms=target_individual_histograms,
        bin_labels=bin_labels
    )

    # plot_movement_signature(movement_signature, bin_labels)
    animate_keypoints(right_hand_source_np, right_hand_targetseg_np)
    visualize_normalized_points(right_hand_source_np, right_hand_target_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Something to do with keypoint analysis, okay!')
    parser.add_argument('--source', type=str, required=True, help='path to the source json file')
    parser.add_argument('--source_video_dir', type=str, required=True, help='source video dir')
    parser.add_argument('--target_video_dir', type=str, required=True, help='target video dir')

    args = parser.parse_args()
    bFound = _main(args.source, args.source_video_dir, args.target_video_dir)
