#!/usr/bin/env python3
"""
Create test video from FLIR thermal images for GUI testing
Converts thermal image sequence to MP4 video
"""
import cv2
import numpy as np
from pathlib import Path
import argparse

def create_thermal_video(
    thermal_dir: str,
    output_path: str = "test_videos/thermal_test.mp4",
    fps: int = 15,
    max_frames: int = 100
):
    """
    Create video from thermal image sequence
    
    Args:
        thermal_dir: Directory containing thermal images
        output_path: Output video path
        fps: Video frame rate
        max_frames: Maximum frames to include
    """
    thermal_path = Path(thermal_dir)
    output_file = Path(output_path)
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get thermal images
    image_files = sorted(list(thermal_path.glob("*.jpg")))[:max_frames]
    
    if not image_files:
        print(f"No thermal images found in {thermal_dir}")
        return
    
    print(f"Found {len(image_files)} thermal images")
    print(f"Creating video: {output_path}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print("Failed to read first image")
        return
    
    height, width = first_image.shape[:2]
    print(f"Video resolution: {width}x{height} @ {fps} fps")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    
    frames_written = 0
    
    for i, image_file in enumerate(image_files):
        # Load thermal image
        thermal_frame = cv2.imread(str(image_file))
        
        if thermal_frame is None:
            print(f"Failed to load {image_file}")
            continue
        
        # Write frame
        writer.write(thermal_frame)
        frames_written += 1
        
        if i % 20 == 0:
            print(f"Processing frame {i+1}/{len(image_files)}")
    
    writer.release()
    
    duration = frames_written / fps
    print(f"✅ Thermal test video created!")
    print(f"📹 File: {output_path}")
    print(f"📊 Frames: {frames_written}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"🎬 Ready for GUI testing!")

def main():
    """Main script"""
    parser = argparse.ArgumentParser(description="Create thermal test video")
    parser.add_argument(
        '--thermal-dir',
        type=str,
        default="datasets/FLIR_ADAS_v2/video_thermal_test/data",
        help='Directory containing thermal images'
    )
    parser.add_argument(
        '--output',
        type=str, 
        default="test_videos/thermal_test.mp4",
        help='Output video path'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Video frame rate'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum frames to include'
    )
    
    args = parser.parse_args()
    
    create_thermal_video(
        thermal_dir=args.thermal_dir,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
