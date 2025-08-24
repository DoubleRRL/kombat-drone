"""
Custom Video Processor for Real-World Detection Testing
Process user-supplied video files with thermal detection and visualization
Like that china street project but for combat scenarios
"""
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from main import MultiModalPipeline
from detection.thermal_yolo import visualize_detections
from advanced_optimizations import LightweightThermalYOLO
from working_thermal_detection import WorkingThermalDetector


class CustomVideoProcessor:
    """
    Process custom video files with real detection and visualization
    Shows bounding boxes, labels, confidence scores like a proper cv project
    """
    
    def __init__(
        self,
        input_video_path: str,
        output_video_path: str = None,
        detection_method: str = "thermal",
        confidence_threshold: float = 0.3,
        show_fps: bool = True
    ):
        self.input_path = Path(input_video_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Video file not found: {input_video_path}")
        
        # Set output path
        if output_video_path:
            self.output_path = Path(output_video_path)
        else:
            stem = self.input_path.stem
            self.output_path = self.input_path.parent / f"{stem}_detected.mp4"
        
        self.detection_method = detection_method
        self.confidence_threshold = confidence_threshold
        self.show_fps = show_fps
        
        # Initialize components
        self.cap = None
        self.writer = None
        self.pipeline = None
        self.detector = None
        
        print(f"🎯 custom video processor ready")
        print(f"   input: {self.input_path}")
        print(f"   output: {self.output_path}")
        print(f"   detection: {detection_method}")
        print(f"   confidence threshold: {confidence_threshold}")
    
    def initialize(self):
        """Initialize video capture and detection pipeline"""
        print("🔧 initializing detection pipeline...")
        
        # Open video file
        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.input_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   video info: {self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps")
        print(f"   total frames: {self.total_frames}")
        
        # Initialize detection method
        if self.detection_method == "thermal":
            self.detector = WorkingThermalDetector()
            print("   using working thermal detector")
        elif self.detection_method == "lightweight":
            self.detector = LightweightThermalYOLO()
            print("   using lightweight yolo")
        else:  # full pipeline
            self.pipeline = MultiModalPipeline()
            print("   using full multimodal pipeline")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        print("✅ initialization complete")
    
    def process_video(self, max_frames: Optional[int] = None):
        """
        Process the entire video with detection and visualization
        """
        if not self.cap:
            self.initialize()
        
        frame_count = 0
        processing_times = []
        detection_counts = []
        
        print(f"🚀 starting video processing...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Process frame
            start_time = time.time()
            processed_frame = self._process_single_frame(frame, frame_count)
            processing_time = (time.time() - start_time) * 1000
            
            processing_times.append(processing_time)
            
            # Write processed frame
            self.writer.write(processed_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:  # every second at 30fps
                progress = (frame_count / self.total_frames) * 100
                avg_time = np.mean(processing_times[-30:])
                current_fps = 1000 / avg_time if avg_time > 0 else 0
                
                print(f"   frame {frame_count}/{self.total_frames} "
                      f"({progress:.1f}%) - {avg_time:.1f}ms/frame ({current_fps:.1f} fps)")
        
        # Final stats
        total_time = sum(processing_times) / 1000  # convert to seconds
        avg_processing_time = np.mean(processing_times)
        avg_fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
        
        print(f"\n✅ video processing complete!")
        print(f"   processed {frame_count} frames in {total_time:.1f}s")
        print(f"   average processing: {avg_processing_time:.1f}ms/frame ({avg_fps:.1f} fps)")
        print(f"   output saved: {self.output_path}")
        
        self.cleanup()
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Process a single frame with detection and visualization
        Returns frame with bounding boxes, labels, and confidence scores
        """
        # Convert to grayscale for thermal processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run detection based on method
        detections = []
        
        if self.detection_method == "thermal":
            # Use working thermal detector
            detections, _ = self.detector.detect_thermal_targets(gray_frame)
        
        elif self.detection_method == "lightweight":
            # Use lightweight YOLO
            detections, _ = self.detector.detect_threats(gray_frame)
        
        else:  # full pipeline
            # Mock IMU data for full pipeline
            imu_data = {
                'accel': [0.0, 0.0, 9.8],
                'gyro': [0.0, 0.0, 0.0]
            }
            
            result = self.pipeline.process_streams(frame, gray_frame, imu_data)
            detections = result.get('detections', [])
        
        # Filter by confidence threshold
        filtered_detections = [
            det for det in detections 
            if hasattr(det, 'confidence') and det.confidence >= self.confidence_threshold
        ]
        
        # Create visualization
        vis_frame = self._create_detection_visualization(
            frame, filtered_detections, frame_idx
        )
        
        return vis_frame
    
    def _create_detection_visualization(
        self, 
        frame: np.ndarray, 
        detections: List, 
        frame_idx: int
    ) -> np.ndarray:
        """
        Create visualization with bounding boxes, labels, and info overlay
        Like that china street project with dozens of boxes and labels
        """
        vis_frame = frame.copy()
        
        # Colors for different classes
        class_colors = {
            'hostile_personnel': (0, 0, 255),      # red
            'hostile_vehicle': (255, 0, 0),       # blue  
            'aircraft_threat': (0, 255, 255),     # yellow
            'equipment': (255, 0, 255),           # magenta
            'thermal_signature': (0, 255, 0),     # green
            'person': (0, 0, 255),                # red
            'car': (255, 0, 0),                   # blue
            'truck': (0, 255, 255),               # yellow
            'bike': (255, 255, 0),                # cyan
            'motorcycle': (128, 0, 128),          # purple
        }
        
        detection_count = 0
        
        for detection in detections:
            if hasattr(detection, 'bbox'):
                x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
                
                # Get class info
                class_name = getattr(detection, 'class_name', 'unknown')
                confidence = getattr(detection, 'confidence', 0.0)
                
                # Get color for this class
                color = class_colors.get(class_name, (255, 255, 255))  # white default
                
                # Draw bounding box (thick for visibility)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
                
                # Create label with class and confidence
                label = f"{class_name}: {confidence:.2f}"
                
                # Get label size for background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw label background
                label_y = max(y1 - 10, label_height + 10)
                cv2.rectangle(
                    vis_frame,
                    (x1, label_y - label_height - baseline - 5),
                    (x1 + label_width + 10, label_y + baseline),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_frame,
                    label,
                    (x1 + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # white text
                    2
                )
                
                # Add thermal info if available
                if hasattr(detection, 'thermal_signature') and detection.thermal_signature:
                    temp_info = detection.thermal_signature
                    temp_text = f"T:{temp_info.get('mean_temp', 0):.1f}°C"
                    cv2.putText(
                        vis_frame,
                        temp_text,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),  # yellow
                        1
                    )
                
                detection_count += 1
        
        # Add info overlay (top-left corner)
        if self.show_fps:
            info_bg_height = 120
            cv2.rectangle(
                vis_frame,
                (10, 10),
                (350, info_bg_height),
                (0, 0, 0),  # black background
                -1
            )
            
            # Add border
            cv2.rectangle(
                vis_frame,
                (10, 10),
                (350, info_bg_height),
                (0, 255, 0),  # green border
                2
            )
            
            # Info text
            info_lines = [
                f"Frame: {frame_idx}",
                f"Detections: {detection_count}",
                f"Method: {self.detection_method}",
                f"Confidence: {self.confidence_threshold:.2f}",
                f"Combat CV Detection System"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 35 + (i * 20)
                color = (0, 255, 0) if i < 4 else (255, 255, 255)
                cv2.putText(
                    vis_frame,
                    line,
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        return vis_frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        if self.pipeline:
            self.pipeline.shutdown()
        cv2.destroyAllWindows()


def main():
    """Main script for processing custom videos"""
    parser = argparse.ArgumentParser(
        description="Process custom video files with combat detection"
    )
    parser.add_argument(
        'input_video',
        type=str,
        help='Path to input video file (mp4, avi, mov, etc.)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output video path (default: input_detected.mp4)'
    )
    parser.add_argument(
        '--method', '-m',
        choices=['thermal', 'lightweight', 'full'],
        default='thermal',
        help='Detection method to use (default: thermal)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.3,
        help='Confidence threshold for detections (default: 0.3)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum frames to process (for testing)'
    )
    parser.add_argument(
        '--no-fps-display',
        action='store_true',
        help='Hide FPS and info overlay'
    )
    
    args = parser.parse_args()
    
    try:
        processor = CustomVideoProcessor(
            input_video_path=args.input_video,
            output_video_path=args.output,
            detection_method=args.method,
            confidence_threshold=args.confidence,
            show_fps=not args.no_fps_display
        )
        
        processor.process_video(max_frames=args.max_frames)
        
    except Exception as e:
        print(f"❌ error processing video: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
