import json
from sklearn.preprocessing import MinMaxScaler
import math
import pickle
import numpy as np

def get_kps_for_range(keypoints, kps_range, one_indexed=True) :
    """ 
    Extract keypoints in range. By default we use 1 based indexing but we can turn it off.
    """
    all_hand_frames = []

    for index in range(len(keypoints)) :
        kp = keypoints[index]
        if one_indexed : 
            hand_kp = [kp[i-1] for i in kps_range] 
        else : 
            hand_kp = [kp[i] for i in kps_range] 
        all_hand_frames.append(hand_kp)
    
    return np.array(all_hand_frames)

def frame_to_timestamp(frame_number, fps):
    """
    Convert a given frame number to a timestamp string (HH:MM:SS.mmm),
    based on the specified frames-per-second (fps).
    """
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def timestamp_to_frame(ts, fps):
    """
    Convert a timestamp string (HH:MM:SS.mmm) to a frame number,
    based on the specified frames-per-second (fps).
    """
    parts = ts.replace(".", ":").split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    if len(parts) > 3:
        seconds += float(parts[3]) / 1000
    total_seconds = hours * 3600 + minutes * 60 + seconds
    frame_number = int(total_seconds * fps)
    return frame_number

def open_keypoints(file_name):
    """
    Load and return pickled keypoints data from the given file.
    """
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_keypoints_for_timerange(keypoints, fps, t1, t2):
    """
    Slice the keypoints array from time t1 to time t2,
    based on the specified frames-per-second (fps).
    Returns the subset of keypoints and the start/end frames.
    """
    kp, kp_sc = keypoints 
    start_frame = int(t1 * fps)
    end_frame = int(t2 * fps)
    return (kp[start_frame:end_frame+1], kp_sc[start_frame:end_frame+1]), start_frame, end_frame

def read_json(json_file):
    """
    Read JSON data from the specified file and return it as a Python object.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def test_timestamp_frame_conversions():
    # Test 1: Zero timestamp and frame
    frame_num = 0
    fps = 30
    ts = frame_to_timestamp(frame_num, fps)
    assert ts == "00:00:00.000", f"Expected '00:00:00.000', got {ts}"
    recovered_frame = timestamp_to_frame(ts, fps)
    assert recovered_frame == frame_num, f"Expected {frame_num}, got {recovered_frame}"

    # Test 2: One second
    frame_num = 30  # 1 second @ 30 fps
    ts = frame_to_timestamp(frame_num, fps)
    assert ts == "00:00:01.000", f"Expected '00:00:01.000', got {ts}"
    recovered_frame = timestamp_to_frame(ts, fps)
    assert recovered_frame == frame_num, f"Expected {frame_num}, got {recovered_frame}"

    # Test 3: Ten seconds
    frame_num = 300  # 10 seconds @ 30 fps
    ts = frame_to_timestamp(frame_num, fps)
    assert ts == "00:00:10.000", f"Expected '00:00:10.000', got {ts}"
    recovered_frame = timestamp_to_frame(ts, fps)
    assert recovered_frame == frame_num, f"Expected {frame_num}, got {recovered_frame}"

    # Test 4: Fractional seconds
    # For example, 45.5 frames -> "00:00:01.517" (approx) when fps=30
    # We'll round frame to int for direct checking.
    fractional_frame = 45.5
    ts = frame_to_timestamp(fractional_frame, fps)
    recovered_frame = timestamp_to_frame(ts, fps)
    # Compare within a tolerance because of floating-point rounding
    assert math.isclose(recovered_frame, int(fractional_frame), abs_tol=1), (
        f"Expected about {fractional_frame}, got {recovered_frame}"
    )

    print("All tests passed for timestamp <-> frame conversions.")

def extract_keypoints_sequence(data):
    """
    Extract keypoints from multiple frames.
    """
    keypoints_sequence = []
    keypoints_sequence_scores = []
    for frame in data:
        if 'predictions' in frame and frame['predictions']:
            if frame['predictions'][0] and 'keypoints' in frame['predictions'][0][0]:
                keypoints = frame['predictions'][0][0]['keypoints']
                keypoints_sequence.append(keypoints)
                keypoints_sequence_scores.append(frame['predictions'][0][0]['keypoint_scores'])
    return np.array(keypoints_sequence), np.array(keypoints_sequence_scores)

def normalise_keypoints(keypoints):
    """
    Normalize the keypoints to the range [0, 1].
    
    Args:
    keypoints (np.array): Array of keypoints of shape (num_frames, num_keypoints, 2)
    
    Returns:
    normalized_keypoints (np.array): Normalized keypoints of the same shape as input.
    """
    keypoints_np = np.array(keypoints)
    flattened_keypoints = keypoints_np.reshape(-1, 2)
    scaler = MinMaxScaler()
    normalized_keypoints = scaler.fit_transform(flattened_keypoints)
    normalized_keypoints = normalized_keypoints.reshape(keypoints_np.shape)
    return normalized_keypoints

if __name__ == "__main__" : 
    test_timestamp_frame_conversions()

