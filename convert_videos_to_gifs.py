#!/usr/bin/env python3
"""
Convert MP4 videos to GIFs for Git hosting
Optimized for size while maintaining quality
"""

import cv2
import os
import glob
from pathlib import Path

def video_to_gif(video_path, output_path, max_width=480, fps=10, quality=85):
    """Convert MP4 to GIF with size optimization"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame skip for target fps
    frame_skip = max(1, int(video_fps / fps))
    
    frames = []
    frame_count = 0
    
    print(f"Converting {video_path} to {output_path}")
    print(f"Original: {total_frames} frames @ {video_fps:.1f} fps")
    print(f"Target: {fps} fps, frame skip: {frame_skip}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Resize frame
            height, width = frame.shape[:2]
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    
    if not frames:
        print(f"Error: No frames extracted from {video_path}")
        return False
    
    print(f"Extracted {len(frames)} frames")
    
    # Save as GIF using PIL
    try:
        from PIL import Image
        import numpy as np
        
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            pil_frame = Image.fromarray(frame)
            pil_frames.append(pil_frame)
        
        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),  # Duration in milliseconds
            loop=0,
            optimize=True,
            quality=quality
        )
        
        # Get file size
        gif_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"GIF saved: {output_path} ({gif_size:.1f} MB)")
        return True
        
    except ImportError:
        print("Error: PIL not available. Install with: pip install Pillow")
        return False
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False

def main():
    """Convert all MP4 videos to GIFs"""
    
    # Create output directories
    output_dirs = {
        'test_videos/euroc_real_slam': 'test_videos/euroc_real_slam_gifs',
        'test_videos/flir_annotated': 'test_videos/flir_annotated_gifs'
    }
    
    for input_dir, output_dir in output_dirs.items():
        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all MP4 files
        mp4_files = glob.glob(os.path.join(input_dir, "*.mp4"))
        
        if not mp4_files:
            print(f"No MP4 files found in {input_dir}")
            continue
            
        print(f"\nProcessing {len(mp4_files)} files in {input_dir}")
        
        for mp4_file in mp4_files:
            filename = os.path.basename(mp4_file)
            name_without_ext = os.path.splitext(filename)[0]
            gif_path = os.path.join(output_dir, f"{name_without_ext}.gif")
            
            # Skip if GIF already exists
            if os.path.exists(gif_path):
                print(f"GIF already exists: {gif_path}")
                continue
            
            # Convert to GIF
            success = video_to_gif(mp4_file, gif_path, max_width=480, fps=8)
            
            if success:
                print(f"✅ Success: {filename} -> {os.path.basename(gif_path)}")
            else:
                print(f"❌ Failed: {filename}")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()
