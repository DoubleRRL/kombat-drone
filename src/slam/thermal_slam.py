"""
Enhanced SLAM Pipeline with Thermal-Visual Sensor Fusion
Integrates ORB-SLAM3 with thermal features using SuperPoint
"""
import numpy as np
import cv2
import torch
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

# Import our fusion modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fusion.sensor_fusion import SensorFusionModule, Feature, PoseWithUncertainty, TrackingState
from fusion.failure_recovery import FailureRecoverySystem


@dataclass
class SLAMConfig:
    """Configuration for thermal SLAM system"""
    # ORB-SLAM3 settings
    vocabulary_path: str = "ORB_SLAM3/Vocabulary/ORBvoc.txt"
    settings_file: str = "config/TUM_RGB-D.yaml"
    
    # SuperPoint settings  
    superpoint_model_path: str = "SuperPointPretrainedNetwork/superpoint_v1.pth"
    superpoint_confidence: float = 0.015
    superpoint_max_keypoints: int = 1000
    
    # Fusion settings
    rgb_weight_good_light: float = 0.8
    thermal_weight_good_light: float = 0.2
    rgb_weight_low_light: float = 0.3
    thermal_weight_low_light: float = 0.7
    illumination_threshold: float = 0.5
    
    # Performance settings
    target_fps: int = 20
    max_processing_time: float = 0.05  # 50ms


class SuperPointExtractor:
    """
    SuperPoint feature extractor for thermal images
    Wraps the pretrained SuperPoint network
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._select_device(device)
        
        # Load SuperPoint model
        try:
            self.net = self._load_superpoint_model()
            print(f"SuperPoint loaded on {self.device}")
        except Exception as e:
            print(f"Failed to load SuperPoint: {e}")
            self.net = None
        
        # Processing parameters
        self.conf_thresh = 0.015
        self.max_keypoints = 1000
        self.nms_dist = 4
        
    def extract_features(self, image: np.ndarray) -> List[Feature]:
        """
        Extract SuperPoint features from thermal image
        
        Args:
            image: Input thermal image (grayscale)
            
        Returns:
            List of Feature objects with keypoints and descriptors
        """
        if self.net is None:
            return self._fallback_extraction(image)
        
        try:
            # Preprocess image
            processed = self._preprocess_image(image)
            
            # Run SuperPoint inference
            with torch.no_grad():
                pred = self.net({'image': processed})
            
            # Extract keypoints and descriptors
            keypoints = pred['keypoints'][0].cpu().numpy()  # [N, 2]
            descriptors = pred['descriptors'][0].cpu().numpy().T  # [N, 256]
            scores = pred['scores'][0].cpu().numpy()  # [N]
            
            # Convert to Feature objects
            features = []
            for i, (kp, desc, score) in enumerate(zip(keypoints, descriptors, scores)):
                if score > self.conf_thresh and len(features) < self.max_keypoints:
                    feature = Feature(
                        pt=kp,
                        descriptor=desc,
                        confidence=float(score),
                        response=float(score)
                    )
                    features.append(feature)
            
            return features
            
        except Exception as e:
            print(f"SuperPoint extraction failed: {e}")
            return self._fallback_extraction(image)
    
    def _load_superpoint_model(self):
        """Load SuperPoint model from checkpoint"""
        # This is a simplified loader - in practice would need the full SuperPoint implementation
        # For now, return None to trigger fallback
        return None
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for SuperPoint"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize to [0, 1]
        normalized = gray.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def _fallback_extraction(self, image: np.ndarray) -> List[Feature]:
        """Fallback feature extraction using OpenCV"""
        # Use ORB as fallback when SuperPoint is not available
        orb = cv2.ORB_create(nfeatures=self.max_keypoints)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        features = []
        if keypoints and descriptors is not None:
            for kp, desc in zip(keypoints, descriptors):
                feature = Feature(
                    pt=np.array([kp.pt[0], kp.pt[1]]),
                    descriptor=desc.astype(np.float32),
                    confidence=kp.response / 100.0,  # Normalize ORB response
                    response=kp.response
                )
                features.append(feature)
        
        return features
    
    def _select_device(self, device: str) -> str:
        """Select optimal device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device


class ORBSLAMWrapper:
    """
    Wrapper for ORB-SLAM3 system
    Handles initialization and pose estimation
    """
    
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.slam_process = None
        self.is_initialized = False
        
        # Mock ORB-SLAM3 for development - replace with actual integration
        self.mock_mode = True
        self.last_pose = None
        
    def initialize(self) -> bool:
        """Initialize ORB-SLAM3 system"""
        try:
            if self.mock_mode:
                print("ORB-SLAM3 initialized in mock mode")
                self.is_initialized = True
                return True
            
            # In production, would launch ORB-SLAM3 process
            # self._launch_orb_slam3()
            
            return self.is_initialized
            
        except Exception as e:
            print(f"Failed to initialize ORB-SLAM3: {e}")
            return False
    
    def track_frame(
        self, 
        rgb_frame: np.ndarray,
        timestamp: float,
        depth_frame: Optional[np.ndarray] = None
    ) -> Optional[PoseWithUncertainty]:
        """
        Track single frame with ORB-SLAM3
        
        Args:
            rgb_frame: RGB image
            timestamp: Frame timestamp
            depth_frame: Optional depth image for RGB-D mode
            
        Returns:
            Pose estimate or None if tracking failed
        """
        if not self.is_initialized:
            return None
        
        if self.mock_mode:
            return self._mock_tracking(rgb_frame, timestamp)
        
        # In production, would interface with actual ORB-SLAM3
        return None
    
    def _mock_tracking(self, rgb_frame: np.ndarray, timestamp: float) -> PoseWithUncertainty:
        """Mock tracking for development"""
        # Simple mock: estimate pose from image center with some noise
        h, w = rgb_frame.shape[:2]
        
        # Add some realistic pose drift
        if self.last_pose is not None:
            # Small random walk
            noise = np.random.normal(0, 0.01, 3)
            position = self.last_pose.position + noise
        else:
            position = np.array([0.0, 0.0, 1.0])
        
        # Mock orientation (identity quaternion with small noise)
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        pose = PoseWithUncertainty(
            position=position,
            orientation=orientation,
            uncertainty=0.1,
            confidence=0.8,
            timestamp=timestamp
        )
        
        self.last_pose = pose
        return pose
    
    def shutdown(self):
        """Shutdown ORB-SLAM3 system"""
        if self.slam_process:
            self.slam_process.terminate()
            self.slam_process = None
        self.is_initialized = False


class ThermalSLAM:
    """
    Main thermal-visual SLAM system
    Integrates ORB-SLAM3 with thermal features and sensor fusion
    """
    
    def __init__(self, config: SLAMConfig):
        self.config = config
        
        # Initialize components
        self.orb_slam = ORBSLAMWrapper(config)
        self.thermal_extractor = SuperPointExtractor(
            config.superpoint_model_path
        )
        self.sensor_fusion = SensorFusionModule()
        self.failure_recovery = FailureRecoverySystem()
        
        # State tracking
        self.tracking_state = TrackingState.GOOD
        self.frame_count = 0
        self.last_successful_pose = None
        
        # Performance monitoring
        self.processing_times = []
        self.slam_stats = {
            'total_frames': 0,
            'successful_tracks': 0,
            'fusion_used': 0,
            'fallback_used': 0
        }
        
        # Initialize system
        self.is_ready = self.orb_slam.initialize()
    
    def process_frame(
        self, 
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        imu_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> PoseWithUncertainty:
        """
        Process synchronized RGB-thermal frame pair
        
        Args:
            rgb_frame: RGB camera image
            thermal_frame: Thermal camera image  
            imu_data: Optional IMU measurements
            timestamp: Frame timestamp
            
        Returns:
            Fused pose estimate
        """
        start_time = time.time()
        timestamp = timestamp or time.time()
        self.frame_count += 1
        self.slam_stats['total_frames'] += 1
        
        try:
            # Step 1: Extract features from both modalities
            rgb_features = self._extract_rgb_features(rgb_frame)
            thermal_features = self.thermal_extractor.extract_features(thermal_frame)
            
            # Step 2: Assess lighting conditions for fusion weighting
            illumination_score = self.sensor_fusion.assess_illumination(rgb_frame)
            thermal_contrast = self.sensor_fusion.assess_thermal_contrast(thermal_frame)
            
            # Step 3: Determine fusion weights based on conditions
            if illumination_score > self.config.illumination_threshold:
                # Good lighting - rely more on RGB
                weights = (self.config.rgb_weight_good_light, 
                          self.config.thermal_weight_good_light)
                primary_features = rgb_features
                secondary_features = thermal_features
            else:
                # Low light - rely more on thermal
                weights = (self.config.rgb_weight_low_light,
                          self.config.thermal_weight_low_light)  
                primary_features = thermal_features
                secondary_features = rgb_features
            
            # Step 4: Cross-modal feature matching
            cross_matches = self.sensor_fusion.match_cross_modal(
                rgb_features, thermal_features
            )
            
            # Step 5: Get ORB-SLAM3 pose estimate
            orb_pose = self.orb_slam.track_frame(rgb_frame, timestamp)
            
            # Step 6: Fuse estimates if ORB-SLAM3 is working
            if orb_pose is not None and orb_pose.confidence > 0.3:
                # Successful ORB-SLAM3 tracking
                fused_pose = self.sensor_fusion.estimate_pose(
                    primary_features, secondary_features, weights, cross_matches
                )
                
                # Blend with ORB-SLAM3 estimate
                fused_pose = self._blend_with_orb_slam(fused_pose, orb_pose)
                self.slam_stats['successful_tracks'] += 1
                self.slam_stats['fusion_used'] += 1
                
            else:
                # ORB-SLAM3 failed - use pure sensor fusion
                fused_pose = self.sensor_fusion.estimate_pose(
                    primary_features, secondary_features, weights, cross_matches
                )
                
                # Validate pose quality
                if not self._validate_pose(fused_pose):
                    fused_pose = self.failure_recovery.handle_tracking_failure(imu_data)
                    self.slam_stats['fallback_used'] += 1
            
            # Step 7: Update failure recovery system
            self.failure_recovery.update_last_good_pose(fused_pose)
            
            # Update tracking state
            if fused_pose.emergency_flag:
                self.tracking_state = TrackingState.EMERGENCY
            elif fused_pose.confidence < 0.3:
                self.tracking_state = TrackingState.LOST
            else:
                self.tracking_state = TrackingState.GOOD
                self.last_successful_pose = fused_pose
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return fused_pose
            
        except Exception as e:
            # Critical failure handling
            print(f"Critical SLAM failure: {e}")
            self.slam_stats['fallback_used'] += 1
            return self.failure_recovery.handle_critical_failure(imu_data, e)
    
    def _extract_rgb_features(self, rgb_frame: np.ndarray) -> List[Feature]:
        """Extract ORB features from RGB frame"""
        # Use ORB for RGB features (could be enhanced with other detectors)
        orb = cv2.ORB_create(nfeatures=1000)
        
        if len(rgb_frame.shape) == 3:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_frame
        
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        features = []
        if keypoints and descriptors is not None:
            for kp, desc in zip(keypoints, descriptors):
                feature = Feature(
                    pt=np.array([kp.pt[0], kp.pt[1]]),
                    descriptor=desc.astype(np.float32),
                    confidence=kp.response / 100.0,
                    response=kp.response
                )
                features.append(feature)
        
        return features
    
    def _blend_with_orb_slam(
        self, 
        fusion_pose: PoseWithUncertainty,
        orb_pose: PoseWithUncertainty
    ) -> PoseWithUncertainty:
        """Blend sensor fusion result with ORB-SLAM3 estimate"""
        # Weight based on confidence levels
        fusion_weight = fusion_pose.confidence
        orb_weight = orb_pose.confidence
        total_weight = fusion_weight + orb_weight
        
        if total_weight > 0:
            fusion_weight /= total_weight
            orb_weight /= total_weight
        else:
            fusion_weight = orb_weight = 0.5
        
        # Blend positions
        blended_position = (
            fusion_weight * fusion_pose.position + 
            orb_weight * orb_pose.position
        )
        
        # Blend orientations using SLERP
        blended_orientation = self.sensor_fusion._slerp_quaternions(
            fusion_pose.orientation, orb_pose.orientation, orb_weight
        )
        
        # Combined uncertainty
        blended_uncertainty = min(fusion_pose.uncertainty, orb_pose.uncertainty)
        
        # Combined confidence  
        blended_confidence = max(fusion_pose.confidence, orb_pose.confidence)
        
        return PoseWithUncertainty(
            position=blended_position,
            orientation=blended_orientation,
            uncertainty=blended_uncertainty,
            confidence=blended_confidence,
            timestamp=fusion_pose.timestamp
        )
    
    def _validate_pose(self, pose: PoseWithUncertainty) -> bool:
        """Validate pose estimate quality"""
        # Basic sanity checks
        if pose.confidence < 0.1:
            return False
        
        if pose.uncertainty > 0.8:
            return False
        
        # Check for reasonable position values
        if np.any(np.abs(pose.position) > 100):  # 100m limit
            return False
        
        # Check for valid quaternion
        quat_norm = np.linalg.norm(pose.orientation)
        if abs(quat_norm - 1.0) > 0.1:
            return False
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get SLAM performance statistics"""
        if self.processing_times:
            avg_time = np.mean(self.processing_times[-100:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            fps = 0
        
        total_frames = self.slam_stats['total_frames']
        success_rate = (
            self.slam_stats['successful_tracks'] / total_frames 
            if total_frames > 0 else 0
        )
        
        return {
            'total_frames_processed': total_frames,
            'tracking_success_rate': success_rate,
            'current_tracking_state': self.tracking_state.value,
            'average_processing_time_ms': avg_time * 1000,
            'processing_fps': fps,
            'fusion_usage_rate': (
                self.slam_stats['fusion_used'] / total_frames 
                if total_frames > 0 else 0
            ),
            'fallback_usage_rate': (
                self.slam_stats['fallback_used'] / total_frames
                if total_frames > 0 else 0  
            ),
            'sensor_fusion_stats': self.sensor_fusion.get_performance_stats(),
            'failure_recovery_stats': self.failure_recovery.get_failure_stats()
        }
    
    def reset(self):
        """Reset SLAM system"""
        self.tracking_state = TrackingState.GOOD
        self.frame_count = 0
        self.last_successful_pose = None
        self.failure_recovery = FailureRecoverySystem()
        
        # Reset ORB-SLAM3
        self.orb_slam.shutdown()
        self.is_ready = self.orb_slam.initialize()
        
        print("Thermal SLAM system reset")
    
    def shutdown(self):
        """Shutdown SLAM system"""
        self.orb_slam.shutdown()
        print("Thermal SLAM system shutdown")
