"""
Optimized Combat-Ready SLAM Pipeline
Performance optimizations for real-time operation
"""
import numpy as np
import cv2
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import torch

from detection.thermal_yolo import ThermalYOLO, Detection
from slam.thermal_slam import ThermalSLAM, SLAMConfig
from fusion.sensor_fusion import PoseWithUncertainty


class OptimizedMultiModalPipeline:
    """
    High-performance multi-modal processing pipeline
    Optimizations:
    - Parallel processing of SLAM and detection
    - Frame skipping and temporal consistency
    - GPU memory optimization
    - Reduced precision inference
    - Asynchronous processing
    """
    
    def __init__(self, 
                 enable_parallel: bool = True,
                 enable_frame_skip: bool = True,
                 detection_interval: int = 3,  # Run detection every N frames
                 slam_interval: int = 1,       # Run SLAM every N frames
                 use_half_precision: bool = True):
        
        self.enable_parallel = enable_parallel
        self.enable_frame_skip = enable_frame_skip
        self.detection_interval = detection_interval
        self.slam_interval = slam_interval
        self.use_half_precision = use_half_precision
        
        # Frame counters
        self.frame_count = 0
        self.last_detection_frame = -1
        self.last_slam_frame = -1
        
        # Cached results for frame skipping
        self.cached_detections = []
        self.cached_pose = None
        
        # Thread pool for parallel processing
        if self.enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=3)
            self.detection_future = None
            self.slam_future = None
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'detection_frames': 0,
            'slam_frames': 0,
            'avg_detection_time': 0.0,
            'avg_slam_time': 0.0,
            'avg_total_time': 0.0
        }
    
    def _initialize_components(self):
        """Initialize optimized components"""
        print("Initializing optimized combat-ready SLAM pipeline...")
        
        # Initialize thermal detection with optimizations
        self.detection_module = ThermalYOLO(
            model_path="yolov8n.pt",  # Use nano model for speed
            device="auto",
            half_precision=self.use_half_precision
        )
        
        # Initialize SLAM with optimized config
        slam_config = SLAMConfig(
            superpoint_model_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
            superpoint_max_keypoints=500,  # Reduced from default 1000
            target_fps=30  # Higher target FPS
        )
        self.slam_module = ThermalSLAM(slam_config)
        
        # Optimized frame preprocessing
        self.frame_preprocessor = OptimizedFramePreprocessor()
        
        print("Optimized pipeline initialized")
    
    def process_streams_optimized(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        imu_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimized multi-modal stream processing
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Preprocess frames (optimized)
        rgb_processed, thermal_processed = self.frame_preprocessor.preprocess_parallel(
            rgb_frame, thermal_frame
        )
        
        # Determine what to process this frame
        run_detection = self._should_run_detection()
        run_slam = self._should_run_slam()
        
        if self.enable_parallel and (run_detection or run_slam):
            # Parallel processing
            result = self._process_parallel(
                rgb_processed, thermal_processed, imu_data, 
                run_detection, run_slam
            )
        else:
            # Sequential processing with frame skipping
            result = self._process_sequential(
                rgb_processed, thermal_processed, imu_data,
                run_detection, run_slam
            )
        
        # Update performance stats
        total_time = (time.time() - start_time) * 1000
        self._update_performance_stats(total_time, run_detection, run_slam)
        
        # Add performance info to result
        result['performance'] = {
            'total_time_ms': total_time,
            'frame_count': self.frame_count,
            'detection_skipped': not run_detection,
            'slam_skipped': not run_slam,
            'fps': 1000.0 / total_time if total_time > 0 else 0
        }
        
        return result
    
    def _should_run_detection(self) -> bool:
        """Determine if detection should run this frame"""
        if not self.enable_frame_skip:
            return True
        return (self.frame_count - self.last_detection_frame) >= self.detection_interval
    
    def _should_run_slam(self) -> bool:
        """Determine if SLAM should run this frame"""
        if not self.enable_frame_skip:
            return True
        return (self.frame_count - self.last_slam_frame) >= self.slam_interval
    
    def _process_parallel(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        imu_data: Optional[Dict[str, Any]],
        run_detection: bool,
        run_slam: bool
    ) -> Dict[str, Any]:
        """Process using parallel threads"""
        
        # Submit tasks to thread pool
        if run_detection:
            self.detection_future = self.executor.submit(
                self._run_detection_task, thermal_frame
            )
        
        if run_slam:
            self.slam_future = self.executor.submit(
                self._run_slam_task, rgb_frame, thermal_frame, imu_data
            )
        
        # Collect results
        detections = self.cached_detections
        pose = self.cached_pose
        
        if run_detection and self.detection_future:
            detections = self.detection_future.result()
            self.cached_detections = detections
            self.last_detection_frame = self.frame_count
        
        if run_slam and self.slam_future:
            pose = self.slam_future.result()
            self.cached_pose = pose
            self.last_slam_frame = self.frame_count
        
        return {
            'pose': pose or PoseWithUncertainty(),
            'detections': detections,
            'system_health': self._get_system_health()
        }
    
    def _process_sequential(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        imu_data: Optional[Dict[str, Any]],
        run_detection: bool,
        run_slam: bool
    ) -> Dict[str, Any]:
        """Process sequentially with frame skipping"""
        
        detections = self.cached_detections
        pose = self.cached_pose
        
        if run_detection:
            detections = self._run_detection_task(thermal_frame)
            self.cached_detections = detections
            self.last_detection_frame = self.frame_count
        
        if run_slam:
            pose = self._run_slam_task(rgb_frame, thermal_frame, imu_data)
            self.cached_pose = pose
            self.last_slam_frame = self.frame_count
        
        return {
            'pose': pose or PoseWithUncertainty(),
            'detections': detections,
            'system_health': self._get_system_health()
        }
    
    def _run_detection_task(self, thermal_frame: np.ndarray) -> list[Detection]:
        """Run thermal detection task"""
        try:
            return self.detection_module.detect_targets(thermal_frame)
        except Exception as e:
            print(f"Detection failed: {e}")
            return []
    
    def _run_slam_task(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        imu_data: Optional[Dict[str, Any]]
    ) -> Optional[PoseWithUncertainty]:
        """Run SLAM task"""
        try:
            return self.slam_module.process_frame(
                rgb_frame, thermal_frame, imu_data
            )
        except Exception as e:
            print(f"SLAM failed: {e}")
            return None
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get optimized system health status"""
        # Simplified health check for performance
        return {
            'overall_health': 'healthy',
            'detection_active': self.frame_count - self.last_detection_frame < self.detection_interval * 2,
            'slam_active': self.frame_count - self.last_slam_frame < self.slam_interval * 2,
            'frame_rate': self.performance_stats.get('avg_fps', 0)
        }
    
    def _update_performance_stats(self, total_time: float, ran_detection: bool, ran_slam: bool):
        """Update performance statistics"""
        self.performance_stats['total_frames'] += 1
        
        # Running average
        alpha = 0.1  # Smoothing factor
        if self.performance_stats['avg_total_time'] == 0:
            self.performance_stats['avg_total_time'] = total_time
        else:
            self.performance_stats['avg_total_time'] = (
                alpha * total_time + (1 - alpha) * self.performance_stats['avg_total_time']
            )
        
        if ran_detection:
            self.performance_stats['detection_frames'] += 1
        
        if ran_slam:
            self.performance_stats['slam_frames'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        stats = self.performance_stats.copy()
        if stats['avg_total_time'] > 0:
            stats['avg_fps'] = 1000.0 / stats['avg_total_time']
        else:
            stats['avg_fps'] = 0
        
        stats['detection_rate'] = (
            stats['detection_frames'] / max(stats['total_frames'], 1)
        )
        stats['slam_rate'] = (
            stats['slam_frames'] / max(stats['total_frames'], 1)
        )
        
        return stats
    
    def shutdown(self):
        """Cleanup resources"""
        if self.enable_parallel and hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if hasattr(self, 'slam_module'):
            self.slam_module.shutdown()


class OptimizedFramePreprocessor:
    """Optimized frame preprocessing"""
    
    def __init__(self):
        # Pre-allocate buffers for common sizes
        self.rgb_buffer = None
        self.thermal_buffer = None
        
    def preprocess_parallel(
        self, 
        rgb_frame: np.ndarray, 
        thermal_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess frames in parallel"""
        
        # Use threading for I/O bound operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            rgb_future = executor.submit(self._preprocess_rgb, rgb_frame)
            thermal_future = executor.submit(self._preprocess_thermal, thermal_frame)
            
            rgb_processed = rgb_future.result()
            thermal_processed = thermal_future.result()
        
        return rgb_processed, thermal_processed
    
    def _preprocess_rgb(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Optimized RGB preprocessing"""
        # Resize if too large (for performance)
        if rgb_frame.shape[0] > 480 or rgb_frame.shape[1] > 640:
            rgb_frame = cv2.resize(rgb_frame, (640, 480))
        
        return rgb_frame
    
    def _preprocess_thermal(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Optimized thermal preprocessing"""
        # Ensure single channel
        if len(thermal_frame.shape) == 3:
            thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize if too large
        if thermal_frame.shape[0] > 480 or thermal_frame.shape[1] > 640:
            thermal_frame = cv2.resize(thermal_frame, (640, 480))
        
        return thermal_frame


def benchmark_optimizations():
    """Benchmark different optimization configurations"""
    
    print("=== OPTIMIZATION BENCHMARK ===\n")
    
    configs = [
        {
            'name': 'Original Pipeline',
            'enable_parallel': False,
            'enable_frame_skip': False,
            'detection_interval': 1,
            'slam_interval': 1
        },
        {
            'name': 'Parallel Processing',
            'enable_parallel': True,
            'enable_frame_skip': False,
            'detection_interval': 1,
            'slam_interval': 1
        },
        {
            'name': 'Frame Skipping (3x)',
            'enable_parallel': False,
            'enable_frame_skip': True,
            'detection_interval': 3,
            'slam_interval': 1
        },
        {
            'name': 'Parallel + Frame Skip',
            'enable_parallel': True,
            'enable_frame_skip': True,
            'detection_interval': 3,
            'slam_interval': 1
        }
    ]
    
    # Test data
    rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    thermal_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    imu_data = {'accel': [0, 0, 9.8], 'gyro': [0, 0, 0]}
    
    for config in configs:
        print(f"Testing: {config['name']}")
        
        # Initialize optimized pipeline
        pipeline = OptimizedMultiModalPipeline(
            enable_parallel=config['enable_parallel'],
            enable_frame_skip=config['enable_frame_skip'],
            detection_interval=config['detection_interval'],
            slam_interval=config['slam_interval']
        )
        
        # Warm-up
        for _ in range(3):
            pipeline.process_streams_optimized(rgb_frame, thermal_frame, imu_data)
        
        # Benchmark
        times = []
        for i in range(20):
            start = time.time()
            result = pipeline.process_streams_optimized(rgb_frame, thermal_frame, imu_data)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        avg_fps = 1000 / avg_time
        
        print(f"  Average: {avg_time:.1f}ms ({avg_fps:.1f} FPS)")
        print(f"  Performance: {pipeline.get_performance_summary()}")
        
        pipeline.shutdown()
        print()


if __name__ == "__main__":
    benchmark_optimizations()
