"""
Video capture demo for README demonstration
Captures system performance with visual output for documentation
"""
import cv2
import numpy as np
import time
from pathlib import Path
import argparse
from typing import Optional

from main import MultiModalPipeline
from detection.thermal_yolo import visualize_detections


class VideoDemo:
    """Create demonstration videos of the SLAM system"""
    
    def __init__(self, output_path: str = "demo_output.mp4", fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self.pipeline = None
        self.video_writer = None
        
    def initialize(self):
        """Initialize the pipeline and video writer"""
        print("Initializing combat-ready SLAM pipeline...")
        self.pipeline = MultiModalPipeline()
        
        # Video writer will be initialized when we know frame size
        self.video_writer = None
        
    def create_demo_video(
        self, 
        duration_seconds: int = 30,
        frame_size: tuple = (1280, 720)
    ):
        """
        Create demonstration video with synthetic data
        
        Args:
            duration_seconds: Length of demo video
            frame_size: Output video resolution
        """
        if not self.pipeline:
            self.initialize()
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, frame_size
        )
        
        total_frames = duration_seconds * self.fps
        print(f"Creating {duration_seconds}s demo video ({total_frames} frames)")
        
        for frame_idx in range(total_frames):
            # Generate synthetic RGB and thermal frames
            rgb_frame, thermal_frame = self._generate_synthetic_frames(
                frame_idx, frame_size
            )
            
            # Mock IMU data
            imu_data = {
                'accel': [0.1 * np.sin(frame_idx * 0.1), 0.0, 9.8],
                'gyro': [0.05 * np.cos(frame_idx * 0.1), 0.0, 0.0]
            }
            
            # Process frame
            start_time = time.time()
            result = self.pipeline.process_streams(rgb_frame, thermal_frame, imu_data)
            processing_time = (time.time() - start_time) * 1000
            
            # Create visualization
            vis_frame = self._create_visualization(
                rgb_frame, thermal_frame, result, processing_time, frame_idx
            )
            
            # Write frame
            self.video_writer.write(vis_frame)
            
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {progress:.1f}% - Frame {frame_idx}/{total_frames}")
        
        self.cleanup()
        print(f"Demo video saved: {self.output_path}")
    
    def create_dataset_video(
        self, 
        dataset_path: str,
        max_frames: int = 100,
        frame_size: tuple = (1280, 720)
    ):
        """
        Create demonstration video using real dataset
        
        Args:
            dataset_path: Path to TUM RGB-D dataset
            max_frames: Maximum frames to process
            frame_size: Output video resolution
        """
        if not self.pipeline:
            self.initialize()
            
        dataset_dir = Path(dataset_path)
        rgb_dir = dataset_dir / "rgb"
        
        if not rgb_dir.exists():
            raise ValueError(f"RGB directory not found: {rgb_dir}")
        
        rgb_files = sorted(list(rgb_dir.glob("*.png")))[:max_frames]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, frame_size
        )
        
        print(f"Creating dataset video with {len(rgb_files)} frames")
        
        for frame_idx, rgb_file in enumerate(rgb_files):
            # Load RGB image
            rgb_frame = cv2.imread(str(rgb_file))
            if rgb_frame is None:
                continue
                
            # Create mock thermal frame
            thermal_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            
            # Mock IMU data
            imu_data = {
                'accel': [0.0, 0.0, 9.8],
                'gyro': [0.0, 0.0, 0.0]
            }
            
            # Process frame
            start_time = time.time()
            result = self.pipeline.process_streams(rgb_frame, thermal_frame, imu_data)
            processing_time = (time.time() - start_time) * 1000
            
            # Create visualization
            vis_frame = self._create_visualization(
                rgb_frame, thermal_frame, result, processing_time, frame_idx
            )
            
            # Resize to target size
            vis_frame = cv2.resize(vis_frame, frame_size)
            
            # Write frame
            self.video_writer.write(vis_frame)
            
            if frame_idx % 10 == 0:
                progress = (frame_idx / len(rgb_files)) * 100
                print(f"  Progress: {progress:.1f}% - Frame {frame_idx}/{len(rgb_files)}")
        
        self.cleanup()
        print(f"Dataset video saved: {self.output_path}")
    
    def _generate_synthetic_frames(
        self, 
        frame_idx: int, 
        frame_size: tuple
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic RGB and thermal frames"""
        height, width = frame_size[1], frame_size[0]
        
        # Generate RGB frame with moving objects
        rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_frame[:] = (50, 80, 120)  # Background
        
        # Add moving "vehicle"
        vehicle_x = int(100 + (frame_idx * 5) % (width - 200))
        vehicle_y = int(height // 2)
        cv2.rectangle(rgb_frame, 
                     (vehicle_x, vehicle_y - 20), 
                     (vehicle_x + 80, vehicle_y + 20), 
                     (0, 255, 0), -1)
        
        # Add moving "person"
        person_x = int(200 + (frame_idx * 3) % (width - 300))
        person_y = int(height // 3)
        cv2.circle(rgb_frame, (person_x, person_y), 15, (255, 0, 0), -1)
        
        # Generate thermal frame (grayscale with heat signatures)
        thermal_frame = np.ones((height, width), dtype=np.uint8) * 100
        
        # Hot vehicle signature
        cv2.rectangle(thermal_frame,
                     (vehicle_x, vehicle_y - 20),
                     (vehicle_x + 80, vehicle_y + 20),
                     200, -1)
        
        # Hot person signature  
        cv2.circle(thermal_frame, (person_x, person_y), 15, 220, -1)
        
        return rgb_frame, thermal_frame
    
    def _create_visualization(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray, 
        result: dict,
        processing_time: float,
        frame_idx: int
    ) -> np.ndarray:
        """Create comprehensive visualization frame"""
        
        # Resize frames for visualization
        vis_height = 320
        vis_width = int(vis_height * rgb_frame.shape[1] / rgb_frame.shape[0])
        
        rgb_vis = cv2.resize(rgb_frame, (vis_width, vis_height))
        thermal_vis = cv2.resize(thermal_frame, (vis_width, vis_height))
        thermal_vis = cv2.applyColorMap(thermal_vis, cv2.COLORMAP_JET)
        
        # Add detections to RGB frame
        detections = result.get('detections', [])
        for det in detections:
            # Scale bounding box to visualization size
            scale_x = vis_width / rgb_frame.shape[1]
            scale_y = vis_height / rgb_frame.shape[0]
            
            x1 = int(det.bbox[0] * scale_x)
            y1 = int(det.bbox[1] * scale_y)
            x2 = int(det.bbox[2] * scale_x)
            y2 = int(det.bbox[3] * scale_y)
            
            cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_vis, f"{det.class_name}: {det.confidence:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Create info panel
        info_panel = np.zeros((vis_height, 300, 3), dtype=np.uint8)
        
        # Add text information
        pose = result.get('pose')
        performance = result.get('performance', {})
        health = result.get('system_health', {})
        
        y_offset = 30
        line_height = 25
        
        texts = [
            f"Frame: {frame_idx}",
            f"Processing: {processing_time:.1f}ms",
            f"FPS: {1000/processing_time:.1f}",
            f"SLAM Conf: {pose.confidence:.2f}" if pose else "SLAM: N/A",
            f"Position: [{pose.position[0]:.1f}, {pose.position[1]:.1f}, {pose.position[2]:.1f}]" if pose else "Pos: N/A",
            f"Detections: {len(detections)}",
            f"Health: {health.get('overall_health', 'unknown')}",
            "",
            "Combat-Ready SLAM",
            "Thermal Detection",
            "Real-time Processing"
        ]
        
        for i, text in enumerate(texts):
            color = (0, 255, 0) if i < 7 else (255, 255, 255)
            cv2.putText(info_panel, text, (10, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Combine all panels
        top_row = np.hstack([rgb_vis, thermal_vis])
        bottom_row = np.hstack([info_panel, np.zeros((vis_height, vis_width * 2 - 300, 3), dtype=np.uint8)])
        
        final_frame = np.vstack([top_row, bottom_row])
        
        return final_frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.video_writer:
            self.video_writer.release()
        if self.pipeline:
            self.pipeline.shutdown()


def main():
    """Main demo script"""
    parser = argparse.ArgumentParser(description="Create demo video")
    parser.add_argument('--output', type=str, default='combat_slam_demo.mp4',
                       help='Output video file')
    parser.add_argument('--duration', type=int, default=30,
                       help='Demo duration in seconds')
    parser.add_argument('--dataset', type=str,
                       help='Path to TUM dataset for real data demo')
    parser.add_argument('--fps', type=int, default=10,
                       help='Output video FPS')
    parser.add_argument('--size', type=str, default='1280x720',
                       help='Output video size (WxH)')
    
    args = parser.parse_args()
    
    # Parse size
    width, height = map(int, args.size.split('x'))
    frame_size = (width, height)
    
    # Create demo
    demo = VideoDemo(args.output, args.fps)
    
    if args.dataset:
        demo.create_dataset_video(args.dataset, frame_size=frame_size)
    else:
        demo.create_demo_video(args.duration, frame_size=frame_size)


if __name__ == "__main__":
    main()
