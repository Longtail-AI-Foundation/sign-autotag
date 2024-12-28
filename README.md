# Autotag 

We aim to automatically detect similar signs across a set of videos in order to create a sign language recognition dataset. This repository is undergoing rapid development.

## Setup 

Install an editable version of this library:

```
python3 -m pip install -e .
```

## Organization

The library is organized as follows:

```
autotag
├── distances # distance functions with a common signature
│   ├── body_range_distance.py
│   ├── dtw_right_hand_distance.py
│   ├── finger_bend_distance.py
│   ├── fingertip_movement_distance.py
│   └── hands_used_distance.py
└── utils # utility functions
    ├── coco_ranges.py
    ├── utils.py
    └── vis_utils.py
```

## Running

```
python3 autotag_runner.py --source ./toshare/sign_metadata/0DHQ4dMCMew_metadata.json --source_video_dir ./toshare/videos/source --target_video_dir ./toshare/videos/target # coffee

python3 autotag_runner.py --source ./toshare/sign_metadata/1vpIZUFKZpQ_metadata.json --source_video_dir ./toshare/videos/source --target_video_dir ./toshare/videos/target # lunch

python3 autotag_runner.py --source ./toshare/sign_metadata/2QqT8tXM920_metadata.json --source_video_dir ./toshare/videos/source --target_video_dir ./toshare/videos/target # age

python3 autotag_runner.py --source ./toshare/sign_metadata/21Y0f3QHoKo_metadata.json --source_video_dir ./toshare/videos/source --target_video_dir ./toshare/videos/target # rupee

python3 autotag_runner.py --source ./toshare/sign_metadata/best-oJL0D3n2e0_metadata.json --source_video_dir ./toshare/videos/source --target_video_dir ./toshare/videos/target # best

python3 autotag_runner.py --source ./toshare/sign_metadata/0REGTy6eqSc_metadata.json --source_video_dir ./toshare/videos/source --target_video_dir ./toshare/videos/target # team
```
