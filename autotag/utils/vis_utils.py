import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# Helper functions to remove repetition

def setup_plot(title, x_label, y_label, fig=None, ax=None, x_lim=None, y_lim=None, figsize=(10, 6)):
    if fig is None and ax is None: 
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig, ax

def plot_line(ax, x_data, y_data, label=None, color="blue", marker=None, linestyle='-', zorder=None):
    ax.plot(x_data, y_data, marker=marker, linestyle=linestyle, color=color, label=label, zorder=zorder)

def plot_vertical_lines(ax, lines_info):
    """
    lines_info: list of tuples (x_position, color, linestyle, label)
    """
    for x_val, color, style, lbl in lines_info:
        ax.axvline(x=x_val, color=color, linestyle=style, label=lbl)

def plot_scatter(ax, x_data, y_data, color="blue", label=None, s=20, zorder=None):
    ax.scatter(x_data, y_data, color=color, label=label, s=s, zorder=zorder)

def plot_bars(ax, x_positions, heights, width=0.35, color='blue', alpha=0.7, label=None):
    ax.bar(x_positions, heights, width=width, color=color, alpha=alpha, label=label)

def finalize_plot(ax, legend=True, show=True):
    if legend:
        ax.legend()
    plt.tight_layout()
    if show: 
        plt.show()

# Refactored functions using the helpers

def animate_keypoints(source_np, target_np):
    num_frames = min(source_np.shape[0], target_np.shape[0])
    source_np = source_np[:num_frames]
    target_np = target_np[:num_frames]

    connections = [
        # Right Hand
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Left Hand
        (21, 22), (22, 23), (23, 24), (24, 25),
        (21, 26), (26, 27), (27, 28), (28, 29),
        (21, 30), (30, 31), (31, 32), (32, 33),
        (21, 34), (34, 35), (35, 36), (36, 37),
        (21, 38), (38, 39), (39, 40), (40, 41),
    ]

    fig, ax = setup_plot(
        title="Keypoints Animation",
        x_label="X-axis",
        y_label="Y-axis",
        x_lim=(0, 1),
        y_lim=(0, 1),
        figsize=(8, 6)
    )

    scatter_source = ax.scatter([], [], c="Blue", s=20, label="Source Keypoints")
    scatter_target = ax.scatter([], [], c="Red", s=20, label="Target Keypoints")

    source_lines = [ax.plot([], [], color="Blue")[0] for _ in connections]
    target_lines = [ax.plot([], [], color="Red")[0] for _ in connections]

    def init():
        scatter_source.set_offsets(np.empty((0, 2)))
        scatter_target.set_offsets(np.empty((0, 2)))
        for line in source_lines + target_lines:
            line.set_data([], [])
        return [scatter_source, scatter_target] + source_lines + target_lines

    def update(frame):
        current_source = source_np[frame]
        current_target = target_np[frame]
        print(f"Frame {frame} - Source Keypoints:\n{current_source}")
        print(f"Frame {frame} - Target Keypoints:\n{current_target}")

        scatter_source.set_offsets(current_source)
        scatter_target.set_offsets(current_target)

        for i, connection in enumerate(connections):
            source_indices = [j for j in connection if j < current_source.shape[0]]
            target_indices = [j for j in connection if j < current_target.shape[0]]

            if len(source_indices) > 1:
                source_lines[i].set_data(current_source[source_indices, 0],
                                         current_source[source_indices, 1])
            else:
                source_lines[i].set_data([], [])

            if len(target_indices) > 1:
                target_lines[i].set_data(current_target[target_indices, 0],
                                         current_target[target_indices, 1])
            else:
                target_lines[i].set_data([], [])

        return [scatter_source, scatter_target] + source_lines + target_lines

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=100, blit=False)
    finalize_plot(ax)

def visualize_normalized_points(source_np, target_np):
    num_frames = min(len(source_np), len(target_np))
    source_np = source_np[:num_frames]
    target_np = target_np[:num_frames]

    fig, ax = setup_plot(
        title="Normalized Keypoints Visualization",
        x_label="X-axis",
        y_label="Y-axis",
        x_lim=(0, 1),
        y_lim=(0, 1),
        figsize=(8, 6)
    )

    scatter_source = ax.scatter([], [], c="blue", s=20, label="Source")
    scatter_target = ax.scatter([], [], c="red", s=20, label="Target")

    def init():
        scatter_source.set_offsets(np.empty((0, 2)))
        scatter_target.set_offsets(np.empty((0, 2)))
        return scatter_source, scatter_target

    def update(frame):
        current_source = source_np[frame]
        current_target = target_np[frame]
        print(f"Frame {frame} - Source: {current_source}")
        print(f"Frame {frame} - Target: {current_target}")
        scatter_source.set_offsets(current_source)
        scatter_target.set_offsets(current_target)
        return scatter_source, scatter_target

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=100, blit=False)
    finalize_plot(ax)

def plot_histograms(source, target, labels, title='Histogram of Source and Target Of Best Sign'):
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = setup_plot(
        title,
        "Categories",
        "Values",
        figsize=(10, 6)
    )
    plot_bars(ax, x - width/2, source, width=width, color='blue', label='Source')
    plot_bars(ax, x + width/2, target, width=width, color='orange', label='Target')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    finalize_plot(ax)

def plot_distance(
    distances,
    title="Distance vs. Window Index",
    x_label="Window Index",
    y_label="Distance",
    figsize=(10, 6),
    # Optional min index/distance point (e.g., for DTW min distance)
    min_index=None,
    min_distance=None,
    # Optional lines for "true" start/end
    true_start=None,
    true_end=None,
    # Whether to show the 'min_index' as a scatter point and line
    show_min=False,
    # Plot style
    color='blue',
    marker=None,
    line_label="Distance",
    fig=None,
    ax=None,
    show=True,
):
    """
    A generic function to plot 'distances' with optional:
      - min_index/min_distance scatter point
      - vertical lines for true_start, true_end
      - vertical line at min_index
    """
    x_values = np.arange(len(distances))

    fig, ax = setup_plot(
        title=title,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        fig=fig,
        ax=ax,
    )

    # Main line
    plot_line(ax, x_values, distances, label=line_label, color=color, marker=marker)

    # Optionally, highlight the minimum distance
    if show_min and min_index is not None and min_distance is not None:
        plot_scatter(ax, [min_index], [min_distance], color="red", label="Min Distance", zorder=5)

    # Build vertical lines
    lines_info = []
    if show_min and min_index is not None:
        lines_info.append((min_index, "red", "--", "Best Match Start"))
    if true_start is not None:
        lines_info.append((true_start, "orange", "--", "True Start Line"))
    if true_end is not None:
        lines_info.append((true_end, "green", "--", "True End Line"))

    # Draw vertical lines if needed
    if lines_info:
        plot_vertical_lines(ax, lines_info)

    finalize_plot(ax, show=show)

def plot_movement_signature(movement_signature, bin_labels):
    x_pos = np.arange(len(movement_signature))
    fig, ax = setup_plot(
        "Hand Movement Signature (Histogram of Directions) OF Rupee Source Sign",
        "Movement Direction",
        "Weighted Movement Magnitude",
        figsize=(8, 6)
    )
    plot_bars(ax, x_pos, movement_signature, width=0.8, color='skyblue', edgecolor='black')
    for i in range(len(movement_signature) + 1):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    finalize_plot(ax)

def plot_movement_signatures_both(source_signature, target_signature, bin_labels):
    x_pos = np.arange(len(bin_labels))
    width = 0.35
    fig, ax = setup_plot(
        "Hand Movement Signature Comparison",
        "Movement Direction",
        "Weighted Movement Magnitude",
        figsize=(8, 6)
    )
    plot_bars(ax, x_pos - width/2, source_signature, width=width, color='skyblue', label='Source Signature')
    plot_bars(ax, x_pos + width/2, target_signature, width=width, color='lightcoral', label='Target Signature')
    for i in range(len(bin_labels) + 1):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    finalize_plot(ax)

def plot_fingertip_histograms(final_histogram, individual_histograms, contributions, bin_labels):
    finger_names = ["Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Little Finger"]
    x_pos = np.arange(len(final_histogram))

    # Final combined histogram
    fig, ax = setup_plot(
        "Final Hand Movement Signature (Combined Histogram)",
        "Movement Direction",
        "Weighted Movement Magnitude",
        figsize=(12, 6)
    )
    plot_bars(ax, x_pos, final_histogram, color='skyblue', edgecolor='black', width=0.8, label="Final Combined Histogram")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    finalize_plot(ax)

    # Individual fingertip histograms
    fig, ax = setup_plot(
        "Individual Movement Signatures for Each Fingertip",
        "Movement Direction",
        "Weighted Movement Magnitude",
        figsize=(12, 8)
    )
    for i, histogram in enumerate(individual_histograms):
        plot_bars(ax, x_pos, histogram, alpha=0.6, label=f"{finger_names[i]} (Contribution: {contributions[i]:.2f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    finalize_plot(ax)

def plot_source_target_histograms(
    source_final_histogram,
    target_final_histogram,
    source_individual_histograms,
    target_individual_histograms,
    bin_labels
):
    finger_names = ["Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Little Finger"]
    x_pos = np.arange(len(bin_labels))

    # Final histograms side by side
    fig, ax = setup_plot(
        "Comparison of Final Combined Histograms (Source vs. Target)",
        "Movement Direction",
        "Weighted Movement Magnitude",
        figsize=(12, 6)
    )
    plot_bars(ax, x_pos - 0.2, source_final_histogram, width=0.4, color='skyblue', label="Source Final Histogram")
    plot_bars(ax, x_pos + 0.2, target_final_histogram, width=0.4, color='orange', label="Target Final Histogram")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    finalize_plot(ax)

    # Individual fingertip histograms
    fig = plt.figure(figsize=(14, 10))
    for i in range(len(finger_names)):
        ax = fig.add_subplot(3, 2, i + 1)
        plot_bars(ax, x_pos - 0.2, source_individual_histograms[i],
                  width=0.4, color='skyblue', alpha=0.7, label=f"Source: {finger_names[i]}")
        plot_bars(ax, x_pos + 0.2, target_individual_histograms[i],
                  width=0.4, color='orange', alpha=0.7, label=f"Target: {finger_names[i]}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.set_xlabel("Movement Direction")
        ax.set_ylabel("Weighted Movement Magnitude")
        ax.set_title(f"{finger_names[i]} Movement Signature")
        ax.legend()
    plt.tight_layout()
    plt.show()

def annotate_and_show_video(video_path, distances, framerate=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    for i, distance in enumerate(distances):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(
            frame,
            f"{distance:.2f}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    ani = FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=1000/framerate,
        blit=True
    )
    plt.show()

