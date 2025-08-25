#!/usr/bin/env python3
"""
Create test video from FLIR thermal images with YOLO detection
Uses trained thermal YOLO model for actual object detection
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from detection.thermal_yolo import ThermalYOLO

def create_thermal_video_with_detection(
    thermal_dir: str,
    model_path: str,
    output_path: str = "test_videos/thermal_test_detected.mp4",
    fps: int = 15,
    max_frames: int = 100,
    confidence_threshold: float = 0.5
):
    """
    Create video from thermal image sequence with YOLO detection
    
    Args:
        thermal_dir: Directory containing thermal images
        model_path: Path to trained YOLO model
        output_path: Output video path
        fps: Video frame rate
        max_frames: Maximum frames to include
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
        print("Make sure the model path is correct and the model is trained")
        return False
    
    # Get thermal images
    image_files = sorted(list(thermal_path.glob("*.jpg")))[:max_frames]
    
    if not image_files:
        print(f"No thermal images found in {thermal_dir}")
        return False
    
    print(f"Found {len(image_files)} thermal images")
    print(f"Creating detected video: {output_path}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print("Failed to read first image")
        return False
    
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
        
        # Run YOLO detection (skip preprocessing for now to avoid opencv errors)
        try:
            detections = detector.detect_targets(thermal_frame, confidence_threshold)
        except Exception as e:
            print(f"Detection failed on frame {i}, using empty detections: {e}")
            detections = []
        
        # Draw detections on frame
        annotated_frame = draw_detections(thermal_frame, detections)
        
        # Add frame info overlay
        info_frame = add_frame_info(annotated_frame, i, len(image_files), len(detections))
        
        # Write frame
        writer.write(info_frame)
        frames_written += 1
        
        if i % 20 == 0:
            print(f"Processing frame {i+1}/{len(image_files)} - {len(detections)} detections")
    
    writer.release()
    
    duration = frames_written / fps
    print(f"✅ Thermal detection video created!")
    print(f"📹 File: {output_path}")
    print(f"📊 Frames: {frames_written}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"🎯 Total detections across all frames")
    
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

def add_frame_info(frame, frame_idx, total_frames, num_detections):
    """Add frame information overlay"""
    overlay = frame.copy()
    
    # Add frame counter
    counter = f"Frame: {frame_idx + 1}/{total_frames}"
    cv2.putText(overlay, counter, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add detection count
    detections_text = f"Detections: {num_detections}"
    cv2.putText(overlay, detections_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add thermal info
    thermal_info = "THERMAL YOLO DETECTION"
    cv2.putText(overlay, thermal_info, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    return overlay

def main():
    """Main script"""
    parser = argparse.ArgumentParser(description="Create thermal test video with YOLO detection")
    parser.add_argument(
        '--thermal-dir',
        type=str,
        default="datasets/FLIR_ADAS_v2/images_thermal_train/data",
        help='Directory containing thermal images'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default="yolov5su.pt",
        help='Path to trained YOLO model'
    )
    parser.add_argument(
        '--output',
        type=str, 
        default="test_videos/thermal_test_detected.mp4",
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
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        print("Available models:")
        print("  - yolov5su.pt (18MB)")
        print("  - yolov8n.pt (6.2MB)")
        return
    
    success = create_thermal_video_with_detection(
        thermal_dir=args.thermal_dir,
        model_path=args.model_path,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence
    )
    
    if success:
        print("🎬 Ready for GUI testing with actual detections!")

if __name__ == "__main__":
    main()
