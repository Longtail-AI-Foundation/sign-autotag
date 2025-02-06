import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from autotag.utils.utils import *
from autotag.utils.vis_utils import *
from autotag.distances.finger_bend_distance import finger_bend_distance

def factorize_close_to_sqrt(n):
    root = int(math.isqrt(n))
    for i in range(root, 0, -1):
        if n % i == 0:
            return i, n // i
    return None, None

def _main(sourcefiles, source_dir, target_dir, distance_fn, hparam_key, hparam_vals, output_dir):
    colors = ['r','g','b','c','m','y','k']

    for sourcefile in sourcefiles:
        # We'll plot all hparam values on the same figure for this file
        fig, ax = plt.subplots()

        # Read source JSON
        source_data = read_json(sourcefile)
        json_basename = os.path.splitext(os.path.basename(sourcefile))[0]
        source_video = os.path.basename(source_data["source"]["source_video"])
        basename = os.path.splitext(source_video)[0]
        source_video = os.path.join(source_dir, source_video)
        source_pkl = os.path.join(source_dir, f"{basename}.pkl")
        word = source_data['source']['source_title']

        source_start_ts = source_data["source"]["signVideos"][0]["signStartTS"]
        source_end_ts = source_data["source"]["signVideos"][0]["signEndTS"]
        source_fps = source_data["source"]["fps"]
        source_pkl_data = open_keypoints(source_pkl)

        # Read target info
        target_video = os.path.basename(source_data["target"]["signVideos"][0]["targetFilePath"])
        basename = os.path.splitext(target_video)[0]
        target_video = os.path.join(target_dir, target_video)
        target_pkl = os.path.join(target_dir, f"{basename}.pkl")
        target_start_ts = source_data["target"]["signVideos"][0]["signStartTS"]
        target_end_ts = source_data["target"]["signVideos"][0]["signEndTS"]
        target_fps = source_data["target"]["signVideos"][0]["fps"]
        target_pkl_data = open_keypoints(target_pkl)

        # Extract keypoints
        kp_source_all = extract_keypoints_sequence(source_pkl_data)
        kp_source_segment, source_start_frame, source_end_frame = get_keypoints_for_timerange(
            kp_source_all, source_fps, source_start_ts, source_end_ts
        )
        kp_target_all = extract_keypoints_sequence(target_pkl_data)
        kp_target_segment, target_start_frame, target_end_frame = get_keypoints_for_timerange(
            kp_target_all, target_fps, target_start_ts, target_end_ts
        )

        # Compute window length and distance
        window_len = target_end_frame - target_start_frame + 1
        print(f'[{json_basename}] Target Window length = {window_len}')

        for j, hparam_val in enumerate(hparam_vals):
            kwargs = {hparam_key: hparam_val}
            distances = distance_fn(kp_source_segment, kp_target_all, window_len, **kwargs)
            
            kwargs = dict()
            if j == 0 :
                kwargs['true_start'] = target_start_frame
                kwargs['true_end'] = target_end_frame

            plot_distance(
                distances,
                line_label=f'{hparam_key}:{hparam_val}',
                fig=fig,
                ax=ax,
                show=False,
                color=colors[j % len(colors)],
                **kwargs
            )

        ax.set_title(word)
        ax.legend()
        plt.tight_layout()

        # Save figure
        out_name = f"{json_basename}_{word}.png"
        plt.savefig(os.path.join(output_dir, out_name))
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distance plotting script.')
    parser.add_argument('--sources', nargs='+', type=str, default=['all'], help='Path(s) to the source json file(s)')
    parser.add_argument('--source_video_dir', type=str, default='./toshare/videos/source', help='Source video dir')
    parser.add_argument('--target_video_dir', type=str, default='./toshare/videos/target', help='Target video dir')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to save plots')

    args = parser.parse_args()
    if len(args.sources) == 1 and args.sources[0] == 'all': 
        sources = list(allFilesWithSuffix('./toshare/sign_metadata', 'json'))
    else: 
        sources = args.sources  

    os.makedirs(args.output_dir, exist_ok=True)

    _main(
        sources,
        args.source_video_dir,
        args.target_video_dir,
        finger_bend_distance,
        'bins',
        [2, 3, 4],
        args.output_dir
    )

