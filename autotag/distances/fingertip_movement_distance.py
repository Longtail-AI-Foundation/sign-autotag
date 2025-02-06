import numpy as np



def fingertip_movement_distance(
    source_kps_and_scores, 
    target_kps_and_scores, 
    window_len, 
    window_stride=1, 
    hand_usage='both', 
    **kwargs
):
    """
    Calculate movement-based distance metric with sliding window.

    Parameters:
        source_kps_and_scores (tuple): Tuple containing source keypoints and scores.
        target_kps_and_scores (tuple): Tuple containing target keypoints and scores.
        window_len (int): Length of the sliding window.
        window_stride (int): Stride of the sliding window.
        hand_usage (str): Specifies hand usage ('right', 'left', or 'both'). Defaults to 'both'.
        **kwargs: Additional keyword arguments.

    Returns:
        numpy.ndarray: Distance matrix for the target sequence.
    """
    source_kps, source_scores = source_kps_and_scores
    target_kps, target_scores = target_kps_and_scores

    # Extract right-hand and left-hand finger keypoints based on hand_usage
    if hand_usage in ['right', 'both']:
        source_right_finger_kps = get_right_finger_keypoints(source_kps)
        target_right_finger_kps = get_right_finger_keypoints(target_kps)
        source_hist_right, _, _ = calculate_movement_signature_allfingertips(*source_right_finger_kps)

    if hand_usage in ['left', 'both']:
        source_left_finger_kps = get_left_finger_keypoints(source_kps)
        target_left_finger_kps = get_left_finger_keypoints(target_kps)
        source_hist_left, _, _ = calculate_movement_signature_allfingertips(*source_left_finger_kps)

    distances = []
    n_frames = len(target_kps)

    for i in range(0, n_frames - window_len + 1, window_stride):
        # Extract sliding window for target
        target_window_kps = target_kps[i:i + window_len]
        target_window_scores = target_scores[i:i + window_len]

        # Calculate movement signature for the target window
        if hand_usage in ['right', 'both']:
            target_hist_right, _, _ = calculate_movement_signature_allfingertips(
                *get_right_finger_keypoints(target_window_kps)
            )
            distance_right = 0.5 * (((source_hist_right - target_hist_right) ** 2) / 
                                   (source_hist_right + target_hist_right + 1e-5)).sum()

        if hand_usage in ['left', 'both']:
            target_hist_left, _, _ = calculate_movement_signature_allfingertips(
                *get_left_finger_keypoints(target_window_kps)
            )
            distance_left = 0.5 * (((source_hist_left - target_hist_left) ** 2) / 
                                  (source_hist_left + target_hist_left + 1e-5)).sum()

        # Combine distances based on hand_usage
        if hand_usage == 'both':
            distance = (distance_right + distance_left) / 2
        elif hand_usage == 'right':
            distance = distance_right
        elif hand_usage == 'left':
            distance = distance_left
        else:
            raise ValueError("Invalid hand_usage. Must be 'right', 'left', or 'both'.")

        distances.append(distance)

    return np.array(distances)


def get_right_finger_keypoints(kp_sequence):
    """
    Extract keypoints for right-hand fingers from the keypoint sequence.

    Parameters:
        kp_sequence (numpy.ndarray): Sequence of keypoints.

    Returns:
        tuple: Keypoints for each finger and the palm-related points.
    """
    right_finger_thumb_kp = get_kps_for_range(kp_sequence, range(116, 117))
    right_middlefing_bottom_kp = get_kps_for_range(kp_sequence, range(121, 122))
    right_hand_bottom_kp = get_kps_for_range(kp_sequence, range(112, 113))
    right_indexfinger_kp = get_kps_for_range(kp_sequence, range(120, 121))
    right_middlefinger_kp = get_kps_for_range(kp_sequence, range(124, 125))
    right_ringfinger_kp = get_kps_for_range(kp_sequence, range(128, 129))
    right_littlefinger_kp = get_kps_for_range(kp_sequence, range(132, 133))

    return (
        right_finger_thumb_kp,
        right_middlefing_bottom_kp,
        right_hand_bottom_kp,
        right_indexfinger_kp,
        right_middlefinger_kp,
        right_ringfinger_kp,
        right_littlefinger_kp
    )


def get_left_finger_keypoints(kp_sequence):
    """
    Extract keypoints for left-hand fingers from the keypoint sequence.

    Parameters:
        kp_sequence (numpy.ndarray): Sequence of keypoints.

    Returns:
        tuple: Keypoints for each finger and the palm-related points.
    """
    left_finger_thumb_kp = get_kps_for_range(kp_sequence, range(95, 96))
    left_middlefing_bottom_kp = get_kps_for_range(kp_sequence, range(100, 101))
    left_hand_bottom_kp = get_kps_for_range(kp_sequence, range(91, 92))
    left_indexfinger_kp = get_kps_for_range(kp_sequence, range(99, 100))  # Adjust range if needed
    left_middlefinger_kp = get_kps_for_range(kp_sequence, range(103, 104))
    left_ringfinger_kp = get_kps_for_range(kp_sequence, range(107, 108))
    left_littlefinger_kp = get_kps_for_range(kp_sequence, range(111, 112))

    return (
        left_finger_thumb_kp,
        left_middlefing_bottom_kp,
        left_hand_bottom_kp,
        left_indexfinger_kp,
        left_middlefinger_kp,
        left_ringfinger_kp,
        left_littlefinger_kp
    )

def get_kps_for_range(kp_sequence, index_range):
    """
    Extract keypoints for a given index range.

    Parameters:
        kp_sequence (numpy.ndarray): Sequence of keypoints.
        index_range (range): Range of indices to extract.

    Returns:
        numpy.ndarray: Extracted keypoints.
    """
    return kp_sequence[:, index_range, :]


# The `calculate_movement_signature_allfingertips` function remains unchanged.
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
            relative_position = frame_diff - palm_curr_frame  # Shape: [2] (vx, vy)

            # Compute movement magnitude and angle
            average_movement = np.mean(relative_position, axis=0)  # Shape: [2] (vx, vy)
            movement_magnitude = np.linalg.norm(average_movement)
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