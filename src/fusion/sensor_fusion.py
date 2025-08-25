"""
Thermal-Visual Sensor Fusion Pipeline
Multi-modal processing combining RGB + thermal streams with cross-validation
"""
import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum


class TrackingState(Enum):
    GOOD = "good"
    LOST = "lost"
    EMERGENCY = "emergency"


@dataclass
class PoseWithUncertainty:
    """Pose estimate with uncertainty bounds"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # quaternion [x, y, z, w] 
    uncertainty: float
    confidence: float
    timestamp: float
    emergency_flag: bool = False


@dataclass
class Feature:
    """Generic feature representation"""
    pt: np.ndarray  # [x, y] pixel coordinates
    descriptor: Optional[np.ndarray] = None
    confidence: float = 1.0
    response: float = 0.0


class SensorFusionModule:
    """
    Core sensor fusion combining RGB and thermal features with geometric validation
    Implements confidence weighting and cross-modal matching
    """
    
    def __init__(self, calibration_matrix: Optional[np.ndarray] = None):
        # Default calibration - should be loaded from file in production
        self.rgb_thermal_calibration = calibration_matrix or np.eye(3)
        self.confidence_threshold = 0.7
        self.spatial_threshold = 10.0  # pixels
        
        # Performance tracking
        self.last_fusion_time = 0.0
        self.fusion_stats = {
            'total_fusions': 0,
            'successful_matches': 0,
            'avg_processing_time': 0.0
        }
    
    def match_cross_modal(
        self, 
        rgb_features: List[Feature], 
        thermal_features: List[Feature],
        confidence_threshold: float = None
    ) -> List[Tuple[Feature, Feature, float]]:
        """
        Match features between RGB and thermal using geometric constraints
        
        Args:
            rgb_features: List of RGB keypoints
            thermal_features: List of thermal keypoints  
            confidence_threshold: Minimum similarity for match
            
        Returns:
            List of (rgb_feat, thermal_feat, similarity) tuples
        """
        start_time = time.time()
        threshold = confidence_threshold or self.confidence_threshold
        
        # Transform thermal features to RGB coordinate system
        thermal_aligned = self._transform_coordinates(
            thermal_features, self.rgb_thermal_calibration
        )
        
        matches = []
        
        # Geometric matching with spatial proximity
        for rgb_feat in rgb_features:
            best_match = None
            best_similarity = 0.0
            
            for thermal_feat in thermal_aligned:
                # Spatial proximity check
                distance = np.linalg.norm(rgb_feat.pt - thermal_feat.pt)
                if distance > self.spatial_threshold:
                    continue
                
                # Compute similarity (geometric + descriptor if available)
                similarity = self._compute_similarity(rgb_feat, thermal_feat, distance)
                
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = thermal_feat
            
            if best_match is not None:
                matches.append((rgb_feat, best_match, best_similarity))
        
        # Update stats
        self.fusion_stats['total_fusions'] += 1
        self.fusion_stats['successful_matches'] += len(matches)
        processing_time = time.time() - start_time
        self._update_avg_time(processing_time)
        
        return matches
    
    def estimate_pose(
        self,
        primary_features: List[Feature],
        secondary_features: List[Feature], 
        weights: Tuple[float, float],
        cross_matches: List[Tuple[Feature, Feature, float]]
    ) -> PoseWithUncertainty:
        """
        Weighted pose estimation using both modalities
        
        Args:
            primary_features: Main feature set (RGB in good light, thermal in low light)
            secondary_features: Supporting feature set
            weights: (primary_weight, secondary_weight) 
            cross_matches: Cross-modal feature correspondences
            
        Returns:
            Fused pose estimate with uncertainty
        """
        # Estimate poses from each modality
        primary_pose = self._estimate_pose_single(primary_features)
        secondary_pose = self._estimate_pose_single(secondary_features)
        
        if primary_pose is None or secondary_pose is None:
            # Fallback to available modality
            return primary_pose or secondary_pose or self._emergency_pose()
        
        # Cross-validation using matched features
        cross_pose = None
        if len(cross_matches) > 20:  # Sufficient cross-matches for validation
            cross_pose = self._estimate_pose_from_matches(cross_matches)
        
        # Weighted fusion
        w1, w2 = weights
        fused_position = w1 * primary_pose.position + w2 * secondary_pose.position
        fused_orientation = self._slerp_quaternions(
            primary_pose.orientation, secondary_pose.orientation, w2
        )
        
        # Cross-match validation bonus
        if cross_pose is not None:
            fused_position = 0.8 * fused_position + 0.2 * cross_pose.position
            fused_orientation = self._slerp_quaternions(
                fused_orientation, cross_pose.orientation, 0.2
            )
        
        # Compute uncertainty based on agreement between modalities
        uncertainty = self._compute_pose_uncertainty(
            primary_pose, secondary_pose, cross_pose
        )
        
        # Overall confidence based on feature quality and cross-validation
        confidence = self._compute_fusion_confidence(
            primary_features, secondary_features, cross_matches
        )
        
        return PoseWithUncertainty(
            position=fused_position,
            orientation=fused_orientation,
            uncertainty=uncertainty,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def assess_illumination(self, rgb_frame: np.ndarray) -> float:
        """Assess lighting conditions in RGB frame"""
        if len(rgb_frame.shape) == 3:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_frame
            
        # Simple brightness metric - could be enhanced with histogram analysis
        mean_brightness = np.mean(gray) / 255.0
        
        # Adjust for contrast - low contrast suggests poor lighting
        contrast = np.std(gray) / 255.0
        
        # Combined illumination score
        illumination_score = (mean_brightness * 0.7 + contrast * 0.3)
        return np.clip(illumination_score, 0.0, 1.0)
    
    def assess_thermal_contrast(self, thermal_frame: np.ndarray) -> float:
        """Assess thermal contrast quality"""
        # Thermal contrast based on temperature variation
        thermal_std = np.std(thermal_frame)
        thermal_range = np.max(thermal_frame) - np.min(thermal_frame)
        
        # Normalize to 0-1 range (assuming 8-bit thermal data)
        contrast_score = min(thermal_std / 50.0, 1.0)  # 50 is empirical threshold
        range_score = min(thermal_range / 255.0, 1.0)
        
        return (contrast_score * 0.6 + range_score * 0.4)
    
    def _transform_coordinates(
        self, 
        features: List[Feature], 
        calibration_matrix: np.ndarray
    ) -> List[Feature]:
        """Transform thermal features to RGB coordinate system"""
        transformed = []
        
        for feat in features:
            # Apply homographic transformation
            pt_homog = np.array([feat.pt[0], feat.pt[1], 1.0])
            transformed_pt = calibration_matrix @ pt_homog
            transformed_pt = transformed_pt[:2] / transformed_pt[2]  # normalize
            
            transformed_feat = Feature(
                pt=transformed_pt,
                descriptor=feat.descriptor,
                confidence=feat.confidence,
                response=feat.response
            )
            transformed.append(transformed_feat)
        
        return transformed
    
    def _compute_similarity(
        self, 
        rgb_feat: Feature, 
        thermal_feat: Feature, 
        spatial_distance: float
    ) -> float:
        """Compute feature similarity combining spatial and descriptor info"""
        # Spatial component (closer = better)
        spatial_sim = max(0.0, 1.0 - spatial_distance / self.spatial_threshold)
        
        # Descriptor component (if available)
        descriptor_sim = 0.5  # default neutral similarity
        if (rgb_feat.descriptor is not None and 
            thermal_feat.descriptor is not None):
            # Cosine similarity for descriptors
            dot_product = np.dot(rgb_feat.descriptor, thermal_feat.descriptor)
            norm_product = (np.linalg.norm(rgb_feat.descriptor) * 
                           np.linalg.norm(thermal_feat.descriptor))
            if norm_product > 0:
                descriptor_sim = (dot_product / norm_product + 1.0) / 2.0  # [0,1]
        
        # Response/confidence component
        response_sim = min(rgb_feat.response, thermal_feat.response)
        confidence_sim = min(rgb_feat.confidence, thermal_feat.confidence)
        
        # Weighted combination
        total_similarity = (
            spatial_sim * 0.4 +
            descriptor_sim * 0.3 + 
            response_sim * 0.2 +
            confidence_sim * 0.1
        )
        
        return total_similarity
    
    def _estimate_pose_single(self, features: List[Feature]) -> Optional[PoseWithUncertainty]:
        """Estimate pose from single modality features"""
        if len(features) < 8:  # Minimum for fundamental matrix
            return None
        
        # Placeholder implementation - would use actual SLAM algorithm
        # For now, return a mock pose based on feature centroid
        centroid = np.mean([f.pt for f in features], axis=0)
        
        return PoseWithUncertainty(
            position=np.array([centroid[0]/100.0, centroid[1]/100.0, 1.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # identity quaternion
            uncertainty=0.1,
            confidence=min(1.0, len(features) / 50.0),
            timestamp=time.time()
        )
    
    def _estimate_pose_from_matches(
        self, 
        matches: List[Tuple[Feature, Feature, float]]
    ) -> Optional[PoseWithUncertainty]:
        """Estimate pose using cross-modal feature matches"""
        if len(matches) < 8:
            return None
        
        # Extract matched points
        rgb_points = np.array([m[0].pt for m in matches])
        thermal_points = np.array([m[1].pt for m in matches])
        
        # Compute fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(
            rgb_points, thermal_points, 
            cv2.FM_RANSAC, 1.0, 0.99
        )
        
        if F is None:
            return None
        
        # Extract pose from fundamental matrix (simplified)
        # In practice, would need camera calibration for proper pose recovery
        centroid = np.mean(rgb_points[mask.ravel() == 1], axis=0)
        
        return PoseWithUncertainty(
            position=np.array([centroid[0]/100.0, centroid[1]/100.0, 1.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            uncertainty=0.05,  # Higher confidence from cross-validation
            confidence=0.9,
            timestamp=time.time()
        )
    
    def _slerp_quaternions(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions"""
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot product is negative, negate one quaternion
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Calculate angle and sin values
        theta_0 = np.arccos(abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        # Compute interpolated quaternion
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def _compute_pose_uncertainty(
        self, 
        pose1: PoseWithUncertainty, 
        pose2: PoseWithUncertainty,
        cross_pose: Optional[PoseWithUncertainty]
    ) -> float:
        """Compute uncertainty based on pose agreement"""
        # Position disagreement
        pos_diff = np.linalg.norm(pose1.position - pose2.position)
        
        # Orientation disagreement (quaternion distance)
        ori_diff = 1.0 - abs(np.dot(pose1.orientation, pose2.orientation))
        
        # Base uncertainty from individual poses
        base_uncertainty = (pose1.uncertainty + pose2.uncertainty) / 2.0
        
        # Increase uncertainty based on disagreement
        disagreement_penalty = pos_diff * 0.1 + ori_diff * 0.2
        
        # Reduce uncertainty if cross-validation agrees
        cross_bonus = 0.0
        if cross_pose is not None:
            cross_pos_diff = np.linalg.norm(pose1.position - cross_pose.position)
            if cross_pos_diff < 0.1:  # Good agreement
                cross_bonus = -0.05
        
        total_uncertainty = base_uncertainty + disagreement_penalty + cross_bonus
        return np.clip(total_uncertainty, 0.01, 1.0)
    
    def _compute_fusion_confidence(
        self,
        primary_features: List[Feature],
        secondary_features: List[Feature], 
        cross_matches: List[Tuple[Feature, Feature, float]]
    ) -> float:
        """Compute overall fusion confidence"""
        # Feature quantity bonus
        feature_score = min(len(primary_features) / 100.0, 1.0) * 0.3
        
        # Feature quality (average response)
        if primary_features:
            quality_score = np.mean([f.response for f in primary_features]) * 0.3
        else:
            quality_score = 0.0
        
        # Cross-validation bonus
        if len(cross_matches) > 0:
            cross_score = min(len(cross_matches) / 50.0, 1.0) * 0.4
        else:
            cross_score = 0.0
        
        total_confidence = feature_score + quality_score + cross_score
        return np.clip(total_confidence, 0.1, 1.0)
    
    def _emergency_pose(self) -> PoseWithUncertainty:
        """Return emergency pose when fusion fails"""
        return PoseWithUncertainty(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            uncertainty=1.0,
            confidence=0.01,
            timestamp=time.time(),
            emergency_flag=True
        )
    
    def _update_avg_time(self, new_time: float):
        """Update running average of processing time"""
        alpha = 0.1  # exponential moving average factor
        if self.fusion_stats['avg_processing_time'] == 0.0:
            self.fusion_stats['avg_processing_time'] = new_time
        else:
            self.fusion_stats['avg_processing_time'] = (
                alpha * new_time + 
                (1 - alpha) * self.fusion_stats['avg_processing_time']
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get fusion performance statistics"""
        total_fusions = self.fusion_stats['total_fusions']
        if total_fusions > 0:
            match_rate = self.fusion_stats['successful_matches'] / total_fusions
        else:
            match_rate = 0.0
        
        return {
            'total_fusions': total_fusions,
            'average_matches_per_fusion': match_rate,
            'average_processing_time_ms': self.fusion_stats['avg_processing_time'] * 1000,
            'fusion_frequency_hz': 1.0 / max(self.fusion_stats['avg_processing_time'], 0.001)
        }
