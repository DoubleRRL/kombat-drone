#!/usr/bin/env python3
"""
Create multiple FLIR thermal videos with actual YOLO detection
Uses trained thermal YOLO model for real object detection
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import random
import os
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from detection.thermal_yolo import ThermalYOLO

def create_flir_video_with_detection(
    thermal_dir: str,
    output_path: str,
    model_path: str = "yolov5su.pt",
    fps: int = 15,
    max_frames: int = 60,
    video_type: str = "general",
    confidence_threshold: float = 0.4
):
    """
    Create FLIR thermal video with actual YOLO detection
    
    Args:
        thermal_dir: Directory containing thermal images
        output_path: Output video path
        model_path: Path to trained YOLO model
        fps: Video frame rate
        max_frames: Maximum frames to include
        video_type: Type of video to create
        confidence_threshold: Detection confidence threshold
    """
    thermal_path = Path(thermal_dir)
    output_file = Path(output_path)
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize thermal YOLO detector
    try:
        detector = ThermalYOLO(model_path=model_path)
        print(f"✅ Loaded thermal YOLO model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load thermal YOLO model: {e}")
        return False
    
    # Get all thermal images
    image_files = sorted(list(thermal_path.glob("*.jpg")))
    
    if not image_files:
        print(f"No thermal images found in {thermal_dir}")
        return False
    
    # Filter images based on video type (using filename patterns)
    if video_type == "cars":
        # Look for images with car-related patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["car", "vehicle", "truck", "bus", "frame-000", "frame-001", "frame-002"])]
    elif video_type == "people":
        # Look for images with people-related patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["person", "pedestrian", "walking", "frame-003", "frame-004", "frame-005"])]
    elif video_type == "driving":
        # Look for driving scene patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["road", "highway", "street", "frame-006", "frame-007", "frame-008"])]
    elif video_type == "urban":
        # Urban environment patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["city", "urban", "building", "frame-009", "frame-010", "frame-011"])]
    elif video_type == "night":
        # Night driving patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["night", "dark", "frame-012", "frame-013", "frame-014"])]
    elif video_type == "highway":
        # Highway driving patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["highway", "freeway", "frame-015", "frame-016", "frame-017"])]
    elif video_type == "intersection":
        # Intersection patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["intersection", "cross", "frame-018", "frame-019", "frame-020"])]
    elif video_type == "parking":
        # Parking lot patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["parking", "lot", "frame-021", "frame-022", "frame-023"])]
    elif video_type == "traffic":
        # Traffic patterns
        filtered_files = [f for f in image_files if any(keyword in f.name.lower() for keyword in 
                       ["traffic", "jam", "frame-024", "frame-025", "frame-026"])]
    else:
        # General/random selection
        filtered_files = image_files
    
    # If filtered list is too small, fall back to random selection
    if len(filtered_files) < max_frames:
        filtered_files = random.sample(image_files, min(len(image_files), max_frames))
    else:
        filtered_files = filtered_files[:max_frames]
    
    print(f"Creating {video_type} video with {len(filtered_files)} frames")
    print(f"Output: {output_path}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(filtered_files[0]))
    if first_image is None:
        print("Failed to read first image")
        return False
    
    height, width = first_image.shape[:2]
    print(f"Video resolution: {width}x{height} @ {fps} fps")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    
    frames_written = 0
    
    for i, image_file in enumerate(filtered_files):
        # Load thermal image
        thermal_frame = cv2.imread(str(image_file))
        
        if thermal_frame is None:
            print(f"Failed to load {image_file}")
            continue
        
        # Run YOLO detection
        detections = detector.detect_targets(thermal_frame, confidence_threshold)
        
        # Draw detections on frame
        annotated_frame = draw_detections(thermal_frame, detections)
        
        # Add video type overlay
        overlay_frame = add_video_overlay(annotated_frame, video_type, i, len(filtered_files), len(detections))
        
        # Write frame
        writer.write(overlay_frame)
        frames_written += 1
        
        if i % 20 == 0:
            print(f"Processing {video_type} frame {i+1}/{len(filtered_files)} - {len(detections)} detections")
    
    writer.release()
    
    duration = frames_written / fps
    print(f"✅ {video_type} video created!")
    print(f"📹 File: {output_path}")
    print(f"📊 Frames: {frames_written}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    
    return True

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    annotated_frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        label = det.class_name
        confidence = det.confidence
        
        # Color based on class
        if 'car' in label.lower() or 'vehicle' in label.lower():
            color = (0, 255, 0)  # Green for vehicles
        elif 'person' in label.lower() or 'pedestrian' in label.lower():
            color = (0, 0, 255)  # Red for people
        else:
            color = (255, 0, 0)  # Blue for other objects
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f"{label} {confidence:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(annotated_frame, 
                     (x1, y1 - label_height - 10), 
                     (x1 + label_width, y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label_text, 
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
    
    return annotated_frame

def add_video_overlay(frame, video_type, frame_idx, total_frames, num_detections):
    """Add video information overlay"""
    overlay = frame.copy()
    
    # Add video type label
    label = f"{video_type.upper()}"
    cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Add frame counter
    counter = f"Frame: {frame_idx + 1}/{total_frames}"
    cv2.putText(overlay, counter, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add detection count
    detections_text = f"Detections: {num_detections}"
    cv2.putText(overlay, detections_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add thermal info
    thermal_info = "THERMAL YOLO DETECTION"
    cv2.putText(overlay, thermal_info, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    return overlay

def main():
    """Main script to create multiple FLIR videos with detection"""
    parser = argparse.ArgumentParser(description="Create multiple FLIR thermal videos with YOLO detection")
    parser.add_argument(
        '--thermal-dir',
        type=str,
        default="datasets/FLIR_ADAS_v2/images_thermal_train/data",
        help='Directory containing thermal images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="test_videos/flir_annotated",
        help='Output directory for annotated videos'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default="yolov5su.pt",
        help='Path to trained YOLO model'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='Video frame rate (lower = slower, clearer for demos)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=150,
        help='Maximum frames per video (150 frames @ 5fps = 30 seconds)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.4,
        help='Detection confidence threshold'
    )
    
    args = parser.parse_args()
    
    # Video types to create
    video_types = [
        "cars", "people", "driving", "urban", "night",
        "highway", "intersection", "parking", "traffic", "general"
    ]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Creating {len(video_types)} FLIR thermal videos with YOLO detection...")
    print(f"📁 Source: {args.thermal_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🤖 Model: {args.model_path}")
    print(f"⚙️  Settings: {args.fps} fps, {args.max_frames} max frames, {args.confidence} confidence")
    print(f"⏱️  Duration: {args.max_frames/args.fps:.1f} seconds per video (slower for clarity)")
    print()
    
    successful_videos = 0
    
    for video_type in video_types:
        output_path = output_dir / f"flir_{video_type}_annotated.mp4"
        
        print(f"🎥 Creating annotated {video_type} video...")
        success = create_flir_video_with_detection(
            thermal_dir=args.thermal_dir,
            output_path=str(output_path),
            model_path=args.model_path,
            fps=args.fps,
            max_frames=args.max_frames,
            video_type=video_type,
            confidence_threshold=args.confidence
        )
        
        if success:
            successful_videos += 1
        
        print("-" * 50)
    
    print(f"🎉 Completed! {successful_videos}/{len(video_types)} annotated videos created successfully")
    print(f"📁 All videos saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
