"""
Main Combat-Ready Low-Light SLAM + Thermal Target Detection Pipeline
Integrates all components for real-time processing
"""
import numpy as np
import cv2
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

from detection.thermal_yolo import ThermalYOLO, visualize_detections
from slam.thermal_slam import ThermalSLAM, SLAMConfig
from fusion.sensor_fusion import PoseWithUncertainty


class SystemHealthMonitor:
    """Monitor overall system health and performance"""
    
    def __init__(self):
        self.sensor_status = {
            'rgb_camera': True,
            'thermal_camera': True, 
            'imu': True
        }
        self.performance_metrics = {
            'fps': 0.0,
            'latency_ms': 0.0,
            'slam_confidence': 0.0,
            'detection_rate': 0.0
        }
        self.alert_thresholds = {
            'min_fps': 15.0,
            'max_latency_ms': 60.0,
            'min_slam_confidence': 0.3
        }
    
    def update_sensor_status(
        self, 
        rgb_frame: Optional[np.ndarray],
        thermal_frame: Optional[np.ndarray], 
        imu_data: Optional[Dict[str, Any]]
    ):
        """Update sensor availability status"""
        self.sensor_status['rgb_camera'] = rgb_frame is not None
        self.sensor_status['thermal_camera'] = thermal_frame is not None
        self.sensor_status['imu'] = imu_data is not None
    
    def update_performance(
        self, 
        fps: float,
        latency_ms: float, 
        slam_confidence: float,
        detection_count: int
    ):
        """Update performance metrics"""
        self.performance_metrics['fps'] = fps
        self.performance_metrics['latency_ms'] = latency_ms
        self.performance_metrics['slam_confidence'] = slam_confidence
        self.performance_metrics['detection_rate'] = detection_count
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        # Check for alerts
        alerts = []
        if self.performance_metrics['fps'] < self.alert_thresholds['min_fps']:
            alerts.append(f"Low FPS: {self.performance_metrics['fps']:.1f}")
        
        if self.performance_metrics['latency_ms'] > self.alert_thresholds['max_latency_ms']:
            alerts.append(f"High latency: {self.performance_metrics['latency_ms']:.1f}ms")
        
        if self.performance_metrics['slam_confidence'] < self.alert_thresholds['min_slam_confidence']:
            alerts.append(f"Low SLAM confidence: {self.performance_metrics['slam_confidence']:.2f}")
        
        # Overall health
        sensors_healthy = all(self.sensor_status.values())
        performance_healthy = len(alerts) == 0
        overall_healthy = sensors_healthy and performance_healthy
        
        return {
            'overall_health': 'healthy' if overall_healthy else 'degraded',
            'sensor_status': self.sensor_status.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'alerts': alerts,
            'timestamp': time.time()
        }


class MultiModalPipeline:
    """
    Main processing pipeline combining SLAM and detection
    Handles multi-modal sensor fusion and real-time processing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        slam_config = SLAMConfig()
        self.slam_module = ThermalSLAM(slam_config)
        self.detection_module = ThermalYOLO()
        self.health_monitor = SystemHealthMonitor()
        
        # Performance tracking
        self.frame_times = []
        self.processing_stats = {
            'frames_processed': 0,
            'average_fps': 0.0,
            'slam_failures': 0,
            'detection_failures': 0
        }
        
        print("Multi-modal pipeline initialized")
    
    def process_streams(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray, 
        imu_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process synchronized sensor streams
        
        Args:
            rgb_frame: RGB camera image
            thermal_frame: Thermal camera image
            imu_data: Optional IMU measurements
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with pose, detections, and system status
        """
        start_time = time.time()
        timestamp = timestamp or start_time
        
        # Update sensor health
        self.health_monitor.update_sensor_status(rgb_frame, thermal_frame, imu_data)
        
        try:
            # Process SLAM and detection in parallel (conceptually)
            # In practice, could use threading for true parallelism
            
            # SLAM processing
            slam_start = time.time()
            pose_result = self.slam_module.process_frame(
                rgb_frame, thermal_frame, imu_data, timestamp
            )
            slam_time = time.time() - slam_start
            
            # Detection processing  
            detection_start = time.time()
            detections = self.detection_module.detect_targets(thermal_frame)
            detection_time = time.time() - detection_start
            
            # Compute performance metrics
            total_time = time.time() - start_time
            fps = 1.0 / total_time if total_time > 0 else 0
            
            # Update performance tracking
            self.frame_times.append(total_time)
            self.processing_stats['frames_processed'] += 1
            
            if len(self.frame_times) > 100:
                self.frame_times = self.frame_times[-100:]  # Keep last 100
            
            avg_time = np.mean(self.frame_times)
            self.processing_stats['average_fps'] = 1.0 / avg_time if avg_time > 0 else 0
            
            # Update health monitor
            self.health_monitor.update_performance(
                fps=fps,
                latency_ms=total_time * 1000,
                slam_confidence=pose_result.confidence,
                detection_count=len(detections)
            )
            
            # Assess fusion quality
            fusion_quality = self._assess_fusion_quality(pose_result, detections)
            
            return {
                'pose': pose_result,
                'detections': detections,
                'system_health': self.health_monitor.get_status(),
                'fusion_quality': fusion_quality,
                'performance': {
                    'total_time_ms': total_time * 1000,
                    'slam_time_ms': slam_time * 1000,
                    'detection_time_ms': detection_time * 1000,
                    'fps': fps
                },
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"Pipeline processing error: {e}")
            self.processing_stats['slam_failures'] += 1
            
            # Return emergency result
            return {
                'pose': PoseWithUncertainty(
                    position=np.array([0, 0, 0]),
                    orientation=np.array([0, 0, 0, 1]),
                    uncertainty=1.0,
                    confidence=0.01,
                    timestamp=timestamp,
                    emergency_flag=True
                ),
                'detections': [],
                'system_health': self.health_monitor.get_status(),
                'fusion_quality': 0.0,
                'error': str(e),
                'timestamp': timestamp
            }
    
    def _assess_fusion_quality(
        self, 
        pose_result: PoseWithUncertainty,
        detections: list
    ) -> float:
        """Assess quality of sensor fusion"""
        quality_factors = []
        
        # Pose quality
        quality_factors.append(pose_result.confidence * 0.4)
        
        # Detection quality
        if detections:
            avg_detection_conf = np.mean([d.confidence for d in detections])
            quality_factors.append(avg_detection_conf * 0.3)
        else:
            quality_factors.append(0.1)  # Penalty for no detections
        
        # System health
        health_status = self.health_monitor.get_status()
        if health_status['overall_health'] == 'healthy':
            quality_factors.append(0.3)
        else:
            quality_factors.append(0.1)
        
        return sum(quality_factors)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'target_fps': 20,
            'max_latency_ms': 50,
            'enable_visualization': True,
            'output_directory': 'results',
            'log_level': 'INFO'
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Failed to load config {config_path}: {e}")
        
        return default_config
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        slam_stats = self.slam_module.get_performance_stats()
        detection_stats = self.detection_module.get_performance_stats()
        health_status = self.health_monitor.get_status()
        
        return {
            'processing_stats': self.processing_stats.copy(),
            'slam_stats': slam_stats,
            'detection_stats': detection_stats,
            'system_health': health_status,
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }
    
    def shutdown(self):
        """Gracefully shutdown pipeline"""
        self.slam_module.shutdown()
        print("Pipeline shutdown complete")


def demo_mode():
    """Run demonstration with sample data"""
    print("Running thermal SLAM demo...")
    
    pipeline = MultiModalPipeline()
    
    # Create mock sensor data
    rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    thermal_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    imu_data = {
        'accel': [0.1, 0.2, 9.8],
        'gyro': [0.01, -0.02, 0.005]
    }
    
    # Process several frames
    for i in range(10):
        print(f"Processing frame {i+1}/10...")
        
        # Add some variation to mock data
        rgb_variation = np.random.randint(-10, 10, rgb_frame.shape, dtype=np.int16)
        thermal_variation = np.random.randint(-5, 5, thermal_frame.shape, dtype=np.int16)
        rgb_frame = np.clip(rgb_frame.astype(np.int16) + rgb_variation, 0, 255).astype(np.uint8)
        thermal_frame = np.clip(thermal_frame.astype(np.int16) + thermal_variation, 0, 255).astype(np.uint8)
        
        # Process frame
        result = pipeline.process_streams(rgb_frame, thermal_frame, imu_data)
        
        # Print results
        pose = result['pose']
        detections = result['detections']
        performance = result['performance']
        
        print(f"  Pose: pos={pose.position}, conf={pose.confidence:.2f}")
        print(f"  Detections: {len(detections)} targets")
        print(f"  Performance: {performance['fps']:.1f} FPS, "
              f"{performance['total_time_ms']:.1f}ms latency")
        
        time.sleep(0.1)  # Simulate processing time
    
    # Print final statistics
    stats = pipeline.get_system_stats()
    print("\nFinal Statistics:")
    print(f"  Frames processed: {stats['processing_stats']['frames_processed']}")
    print(f"  Average FPS: {stats['processing_stats']['average_fps']:.1f}")
    print(f"  System health: {stats['system_health']['overall_health']}")
    
    pipeline.shutdown()


def process_dataset(dataset_path: str):
    """Process dataset sequences"""
    print(f"Processing dataset: {dataset_path}")
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        return
    
    pipeline = MultiModalPipeline()
    
    # Look for TUM RGB-D format
    rgb_dir = dataset_dir / "rgb"
    depth_dir = dataset_dir / "depth"
    
    if rgb_dir.exists():
        print("Found TUM RGB-D dataset format")
        
        # Get RGB images
        rgb_files = sorted(list(rgb_dir.glob("*.png")))
        
        for i, rgb_file in enumerate(rgb_files[:50]):  # Process first 50 frames
            print(f"Processing frame {i+1}/{min(50, len(rgb_files))}: {rgb_file.name}")
            
            # Load RGB image
            rgb_frame = cv2.imread(str(rgb_file))
            if rgb_frame is None:
                continue
            
            # Create mock thermal frame (would load actual thermal in production)
            thermal_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            
            # Mock IMU data
            imu_data = {
                'accel': [0.0, 0.0, 9.8],
                'gyro': [0.0, 0.0, 0.0]
            }
            
            # Process frame
            result = pipeline.process_streams(rgb_frame, thermal_frame, imu_data)
            
            if i % 10 == 0:  # Print every 10th frame
                pose = result['pose']
                print(f"  Frame {i}: confidence={pose.confidence:.2f}, "
                      f"fps={result['performance']['fps']:.1f}")
    
    else:
        print("No recognized dataset format found")
    
    # Print final statistics
    stats = pipeline.get_system_stats()
    print(f"\nDataset processing complete:")
    print(f"  Total frames: {stats['processing_stats']['frames_processed']}")
    print(f"  Average FPS: {stats['processing_stats']['average_fps']:.1f}")
    
    pipeline.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Combat-Ready Low-Light SLAM + Thermal Target Detection"
    )
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run demonstration mode with mock data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset directory to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mode()
    elif args.dataset:
        process_dataset(args.dataset)
    else:
        print("Please specify --demo or --dataset. Use --help for more options.")


if __name__ == "__main__":
    main()
