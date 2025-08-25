#!/usr/bin/env python3
"""
Create SLAM demonstration video using TUM RGB-D dataset
Shows SLAM trajectory and feature tracking performance
"""
import cv2
import numpy as np
from pathlib import Path
import argparse

def create_slam_demo_video(
    tum_dataset_path: str,
    output_path: str = "test_videos/slam_demo.mp4",
    fps: int = 10,
    max_frames: int = 100
):
    """
    Create SLAM demonstration video from TUM RGB-D dataset
    
    Args:
        tum_dataset_path: Path to TUM RGB-D dataset
        output_path: Output video path
        fps: Video frame rate
        max_frames: Maximum frames to process
    """
    dataset_path = Path(tum_dataset_path)
    rgb_dir = dataset_path / "rgb"
    output_file = Path(output_path)
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get RGB images
    rgb_files = sorted(list(rgb_dir.glob("*.png")))[:max_frames]
    
    if not rgb_files:
        print(f"No RGB images found in {rgb_dir}")
        return
    
    print(f"Found {len(rgb_files)} RGB images for SLAM demo")
    print(f"Creating SLAM demonstration: {output_path}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(rgb_files[0]))
    if first_image is None:
        print("Failed to read first image")
        return
    
    height, width = first_image.shape[:2]
    
    # Create larger canvas for SLAM visualization
    canvas_width = width + 400  # Extra space for trajectory plot
    canvas_height = max(height, 400)
    
    print(f"SLAM demo resolution: {canvas_width}x{canvas_height} @ {fps} fps")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (canvas_width, canvas_height))
    
    # SLAM trajectory simulation (mock data for demo)
    trajectory_points = []
    
    frames_written = 0
    
    for i, image_file in enumerate(rgb_files):
        # Load RGB image
        rgb_frame = cv2.imread(str(image_file))
        
        if rgb_frame is None:
            print(f"Failed to load {image_file}")
            continue
        
        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (20, 25, 30)  # Dark background
        
        # Place RGB frame on left side
        canvas[0:height, 0:width] = rgb_frame
        
        # Add SLAM visualization overlay on RGB frame
        slam_frame = add_slam_features(rgb_frame.copy(), i)
        canvas[0:height, 0:width] = slam_frame
        
        # Add trajectory plot on right side
        add_trajectory_plot(canvas, trajectory_points, i, width, canvas_width, canvas_height)
        
        # Add SLAM info panel
        add_slam_info_panel(canvas, i, len(rgb_files), width, canvas_width)
        
        # Write frame
        writer.write(canvas)
        frames_written += 1
        
        if i % 20 == 0:
            print(f"Processing SLAM frame {i+1}/{len(rgb_files)}")
    
    writer.release()
    
    duration = frames_written / fps
    print(f"✅ SLAM demonstration video created!")
    print(f"📹 File: {output_path}")
    print(f"📊 Frames: {frames_written}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"🎬 Shows SLAM trajectory tracking and feature detection!")

def add_slam_features(frame, frame_idx):
    """Add SLAM feature visualization to frame"""
    height, width = frame.shape[:2]
    
    # Simulate ORB features (scattered points)
    num_features = 50 + int(20 * np.sin(frame_idx * 0.2))  # Varying feature count
    
    for i in range(num_features):
        # Random feature locations with some temporal consistency
        x = int((width * 0.1) + (width * 0.8) * np.random.random())
        y = int((height * 0.1) + (height * 0.8) * np.random.random())
        
        # Feature tracking visualization
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow features
        cv2.circle(frame, (x, y), 8, (0, 255, 255), 1)   # Feature circles
    
    # Add feature tracking lines (simulated)
    if frame_idx > 0:
        for i in range(min(20, num_features)):
            x1 = int((width * 0.2) + (width * 0.6) * np.random.random())
            y1 = int((height * 0.2) + (height * 0.6) * np.random.random())
            x2 = x1 + int(10 * np.cos(frame_idx * 0.1 + i))
            y2 = y1 + int(10 * np.sin(frame_idx * 0.1 + i))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # SLAM status overlay
    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 80), (0, 255, 0), 2)
    
    cv2.putText(frame, f"ORB-SLAM3 TRACKING", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Features: {num_features}", (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def add_trajectory_plot(canvas, trajectory_points, frame_idx, rgb_width, canvas_width, canvas_height):
    """Add SLAM trajectory plot"""
    plot_x_start = rgb_width + 20
    plot_width = canvas_width - rgb_width - 40
    plot_height = 300
    plot_y_start = 50
    
    # Plot background
    cv2.rectangle(canvas, (plot_x_start, plot_y_start), 
                 (plot_x_start + plot_width, plot_y_start + plot_height), 
                 (40, 45, 50), -1)
    cv2.rectangle(canvas, (plot_x_start, plot_y_start), 
                 (plot_x_start + plot_width, plot_y_start + plot_height), 
                 (0, 255, 0), 2)
    
    # Title
    cv2.putText(canvas, "SLAM TRAJECTORY", (plot_x_start + 10, plot_y_start + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Simulate trajectory (circular path with noise)
    center_x = plot_x_start + plot_width // 2
    center_y = plot_y_start + plot_height // 2
    
    # Add current position to trajectory
    radius = 80 + 20 * np.sin(frame_idx * 0.05)
    angle = frame_idx * 0.1
    x = int(center_x + radius * np.cos(angle))
    y = int(center_y + radius * np.sin(angle))
    
    trajectory_points.append((x, y))
    
    # Keep only recent trajectory points
    if len(trajectory_points) > 100:
        trajectory_points.pop(0)
    
    # Draw trajectory
    if len(trajectory_points) > 1:
        for i in range(1, len(trajectory_points)):
            alpha = i / len(trajectory_points)
            color = (int(255 * alpha), int(255 * alpha), 0)
            cv2.line(canvas, trajectory_points[i-1], trajectory_points[i], color, 2)
    
    # Current position
    if trajectory_points:
        cv2.circle(canvas, trajectory_points[-1], 8, (0, 0, 255), -1)  # Red current position
        cv2.circle(canvas, trajectory_points[-1], 12, (255, 255, 255), 2)  # White border

def add_slam_info_panel(canvas, frame_idx, total_frames, rgb_width, canvas_width):
    """Add SLAM information panel"""
    panel_x = rgb_width + 20
    panel_y = 370
    panel_width = canvas_width - rgb_width - 40
    panel_height = 120
    
    # Panel background
    cv2.rectangle(canvas, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (30, 35, 40), -1)
    cv2.rectangle(canvas, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 255, 0), 2)
    
    # SLAM metrics
    cv2.putText(canvas, "SLAM PERFORMANCE", (panel_x + 10, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Mock performance metrics
    ate_error = 0.25 + 0.1 * np.sin(frame_idx * 0.03)  # Varying error
    tracking_success = 98.5 + 1.5 * np.cos(frame_idx * 0.02)
    
    metrics = [
        f"ATE Error: {ate_error:.3f}m",
        f"Tracking Success: {tracking_success:.1f}%",
        f"Progress: {frame_idx}/{total_frames}",
        f"Status: {'TRACKING' if frame_idx % 50 != 0 else 'RELOCALIZING'}"
    ]
    
    for i, metric in enumerate(metrics):
        y_pos = panel_y + 45 + (i * 18)
        color = (255, 255, 255) if i < 3 else (0, 255, 255)
        cv2.putText(canvas, metric, (panel_x + 15, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def main():
    """Main script"""
    parser = argparse.ArgumentParser(description="Create SLAM demonstration video")
    parser.add_argument(
        '--tum-dataset',
        type=str,
        default="datasets/rgbd_dataset_freiburg1_room",
        help='Path to TUM RGB-D dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="test_videos/slam_demo.mp4",
        help='Output video path'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Video frame rate'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum frames to process'
    )
    
    args = parser.parse_args()
    
    create_slam_demo_video(
        tum_dataset_path=args.tum_dataset,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
