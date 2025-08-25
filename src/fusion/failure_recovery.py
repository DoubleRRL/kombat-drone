"""
Failure Recovery System for Sensor Fusion Pipeline
Handles graceful degradation, IMU dead reckoning, and emergency modes
"""
import numpy as np
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fusion.sensor_fusion import PoseWithUncertainty, TrackingState


@dataclass
class EmergencyPose:
    """Emergency pose for hover-in-place mode"""
    position: np.ndarray
    orientation: np.ndarray  
    confidence: float
    emergency_flag: bool = True


class IMUIntegrator:
    """Dead reckoning using IMU data when visual tracking fails"""
    
    def __init__(self):
        self.last_pose = None
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.integration_start_time = None
        
        # IMU noise parameters (should be calibrated)
        self.accel_noise_std = 0.01  # m/s^2
        self.gyro_noise_std = 0.001  # rad/s
        
    def integrate(self, imu_data: Dict[str, Any], dt: float = None) -> np.ndarray:
        """
        Integrate IMU measurements to estimate pose change
        
        Args:
            imu_data: Dictionary with 'accel' and 'gyro' measurements
            dt: Time step (auto-computed if None)
            
        Returns:
            Estimated pose change [dx, dy, dz, dqx, dqy, dqz, dqw]
        """
        if dt is None:
            current_time = time.time()
            if self.integration_start_time is None:
                self.integration_start_time = current_time
                dt = 0.033  # Default 30Hz
            else:
                dt = current_time - self.integration_start_time
                self.integration_start_time = current_time
        
        # Extract measurements
        accel = np.array(imu_data.get('accel', [0, 0, 0]))
        gyro = np.array(imu_data.get('gyro', [0, 0, 0]))
        
        # Add noise simulation for realistic uncertainty
        accel += np.random.normal(0, self.accel_noise_std, 3)
        gyro += np.random.normal(0, self.gyro_noise_std, 3)
        
        # Remove gravity (assuming IMU is roughly level)
        accel[2] -= 9.81
        
        # Integrate acceleration to get velocity change
        dv = accel * dt
        self.velocity += dv
        
        # Integrate velocity to get position change
        dp = self.velocity * dt + 0.5 * accel * dt**2
        
        # Integrate angular velocity to get orientation change
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-6:
            axis = gyro / np.linalg.norm(gyro)
            dq = self._axis_angle_to_quaternion(axis, angle)
        else:
            dq = np.array([0, 0, 0, 1])  # identity quaternion
        
        return np.concatenate([dp, dq])
    
    def reset(self, initial_pose: PoseWithUncertainty):
        """Reset integrator with known good pose"""
        self.last_pose = initial_pose
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.integration_start_time = None
    
    def _axis_angle_to_quaternion(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion representation"""
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)
        
        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half, 
            axis[2] * sin_half,
            cos_half
        ])


class FailureRecoverySystem:
    """
    Hierarchical failure recovery for sensor fusion pipeline
    Implements graceful degradation and emergency modes
    """
    
    def __init__(self):
        self.last_good_pose = None
        self.failure_start_time = None
        self.tracking_state = TrackingState.GOOD
        
        self.imu_integrator = IMUIntegrator()
        self.relocalization_attempts = 0
        self.max_relocalization_attempts = 3
        self.max_imu_fallback_time = 5.0  # seconds
        
        # Failure statistics
        self.failure_stats = {
            'total_failures': 0,
            'imu_recoveries': 0,
            'relocalization_successes': 0,
            'emergency_activations': 0
        }
    
    def handle_tracking_failure(
        self, 
        imu_data: Optional[Dict[str, Any]] = None
    ) -> PoseWithUncertainty:
        """
        Handle visual tracking failure with hierarchical recovery
        
        Args:
            imu_data: IMU measurements for dead reckoning
            
        Returns:
            Best available pose estimate
        """
        if self.failure_start_time is None:
            self.failure_start_time = time.time()
            self.failure_stats['total_failures'] += 1
        
        self.tracking_state = TrackingState.LOST
        
        # Option 1: IMU dead reckoning (short term)
        if self.can_use_imu_fallback() and imu_data is not None:
            return self._imu_propagate(imu_data)
        
        # Option 2: Attempt relocalization
        elif self._should_attempt_relocalization():
            relocalization_result = self._attempt_relocalization()
            if relocalization_result is not None:
                return relocalization_result
        
        # Option 3: Emergency hover mode
        return self._emergency_mode()
    
    def handle_critical_failure(
        self, 
        imu_data: Optional[Dict[str, Any]] = None,
        exception: Exception = None
    ) -> PoseWithUncertainty:
        """Handle critical system failures"""
        self.tracking_state = TrackingState.EMERGENCY
        self.failure_stats['emergency_activations'] += 1
        
        print(f"CRITICAL FAILURE: {exception}")
        
        # Try IMU fallback even in critical mode
        if imu_data is not None and self.last_good_pose is not None:
            try:
                return self._imu_propagate(imu_data)
            except Exception:
                pass  # Fall through to emergency
        
        return self._emergency_mode()
    
    def can_use_imu_fallback(self) -> bool:
        """Check if IMU dead reckoning is still viable"""
        if self.failure_start_time is None:
            return False
        
        time_since_failure = time.time() - self.failure_start_time
        return time_since_failure < self.max_imu_fallback_time
    
    def update_last_good_pose(self, pose: PoseWithUncertainty):
        """Update reference pose for failure recovery"""
        if not pose.emergency_flag and pose.confidence > 0.5:
            self.last_good_pose = pose
            self.imu_integrator.reset(pose)
            
            # Reset failure state if tracking is good
            if self.tracking_state != TrackingState.GOOD:
                self._reset_failure_state()
    
    def _imu_propagate(self, imu_data: Dict[str, Any]) -> PoseWithUncertainty:
        """Dead reckoning using IMU data"""
        if self.last_good_pose is None:
            return self._emergency_mode()
        
        try:
            # Integrate IMU motion since last good pose
            delta_pose = self.imu_integrator.integrate(imu_data)
            
            # Apply delta to last good pose
            estimated_position = self.last_good_pose.position + delta_pose[:3]
            estimated_orientation = self._multiply_quaternions(
                self.last_good_pose.orientation, delta_pose[3:]
            )
            
            # Compute growing uncertainty over time
            time_since_failure = time.time() - self.failure_start_time
            uncertainty = self._compute_growing_uncertainty(time_since_failure)
            
            # Decreasing confidence over time
            confidence = max(0.1, 0.8 * np.exp(-time_since_failure / 2.0))
            
            self.failure_stats['imu_recoveries'] += 1
            
            return PoseWithUncertainty(
                position=estimated_position,
                orientation=estimated_orientation,
                uncertainty=uncertainty,
                confidence=confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"IMU propagation failed: {e}")
            return self._emergency_mode()
    
    def _should_attempt_relocalization(self) -> bool:
        """Check if relocalization should be attempted"""
        return self.relocalization_attempts < self.max_relocalization_attempts
    
    def _attempt_relocalization(self) -> Optional[PoseWithUncertainty]:
        """
        Try to re-initialize SLAM with relaxed parameters
        This is a placeholder - would integrate with actual SLAM system
        """
        self.relocalization_attempts += 1
        
        # Simulate relocalization attempt
        # In practice, this would:
        # 1. Reduce feature matching thresholds
        # 2. Expand search regions  
        # 3. Try different initialization strategies
        
        relocalization_success_rate = 0.6  # Empirical success rate
        if np.random.random() < relocalization_success_rate:
            # Successful relocalization
            self.failure_stats['relocalization_successes'] += 1
            
            # Return a pose near the last known position with high uncertainty
            if self.last_good_pose is not None:
                noise = np.random.normal(0, 0.5, 3)  # 0.5m position noise
                estimated_position = self.last_good_pose.position + noise
            else:
                estimated_position = np.array([0.0, 0.0, 1.0])
            
            return PoseWithUncertainty(
                position=estimated_position,
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                uncertainty=0.8,  # High uncertainty after relocalization
                confidence=0.6,   # Moderate confidence
                timestamp=time.time()
            )
        
        return None  # Relocalization failed
    
    def _emergency_mode(self) -> PoseWithUncertainty:
        """Last resort - return hover-in-place command"""
        self.tracking_state = TrackingState.EMERGENCY
        
        # Use last known position or origin
        if self.last_good_pose is not None:
            emergency_position = self.last_good_pose.position.copy()
        else:
            emergency_position = np.array([0.0, 0.0, 1.0])  # 1m altitude hover
        
        return PoseWithUncertainty(
            position=emergency_position,
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # Level hover
            uncertainty=1.0,     # Maximum uncertainty
            confidence=0.01,     # Minimal confidence
            timestamp=time.time(),
            emergency_flag=True
        )
    
    def _reset_failure_state(self):
        """Reset failure recovery state when tracking resumes"""
        self.failure_start_time = None
        self.relocalization_attempts = 0
        self.tracking_state = TrackingState.GOOD
        print("Tracking recovered - failure state reset")
    
    def _compute_growing_uncertainty(self, time_since_failure: float) -> float:
        """Compute uncertainty that grows over time during IMU fallback"""
        # Exponential growth in uncertainty
        base_uncertainty = 0.1
        growth_rate = 0.2  # uncertainty doubles every 5 seconds
        
        uncertainty = base_uncertainty * np.exp(growth_rate * time_since_failure)
        return min(uncertainty, 1.0)  # Cap at maximum uncertainty
    
    def _multiply_quaternions(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (Hamilton product)"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure recovery statistics"""
        total_failures = self.failure_stats['total_failures']
        
        if total_failures > 0:
            imu_success_rate = self.failure_stats['imu_recoveries'] / total_failures
            reloc_success_rate = (self.failure_stats['relocalization_successes'] / 
                                max(self.relocalization_attempts, 1))
        else:
            imu_success_rate = 0.0
            reloc_success_rate = 0.0
        
        return {
            'total_failures': total_failures,
            'current_state': self.tracking_state.value,
            'imu_success_rate': imu_success_rate,
            'relocalization_success_rate': reloc_success_rate,
            'emergency_activations': self.failure_stats['emergency_activations'],
            'time_since_last_failure': (
                time.time() - self.failure_start_time 
                if self.failure_start_time else None
            )
        }
