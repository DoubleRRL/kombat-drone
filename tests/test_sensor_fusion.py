"""
Tests for sensor fusion module
Validates cross-modal matching and pose estimation
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fusion.sensor_fusion import SensorFusionModule, Feature, PoseWithUncertainty


class TestSensorFusionModule:
    """Test cases for sensor fusion functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fusion_module = SensorFusionModule()
        
        # Create mock features
        self.rgb_features = [
            Feature(pt=np.array([100.0, 150.0]), confidence=0.8, response=0.9),
            Feature(pt=np.array([200.0, 250.0]), confidence=0.7, response=0.8),
            Feature(pt=np.array([300.0, 350.0]), confidence=0.9, response=0.95)
        ]
        
        self.thermal_features = [
            Feature(pt=np.array([101.0, 151.0]), confidence=0.75, response=0.85),
            Feature(pt=np.array([201.0, 251.0]), confidence=0.8, response=0.9),
            Feature(pt=np.array([301.0, 351.0]), confidence=0.85, response=0.88)
        ]
    
    def test_cross_modal_matching(self):
        """Test cross-modal feature matching"""
        matches = self.fusion_module.match_cross_modal(
            self.rgb_features, self.thermal_features
        )
        
        # Should find matches for close features
        assert len(matches) > 0
        assert len(matches) <= min(len(self.rgb_features), len(self.thermal_features))
        
        # Check match structure
        for rgb_feat, thermal_feat, similarity in matches:
            assert isinstance(rgb_feat, Feature)
            assert isinstance(thermal_feat, Feature)
            assert 0.0 <= similarity <= 1.0
    
    def test_pose_estimation(self):
        """Test pose estimation from features"""
        matches = self.fusion_module.match_cross_modal(
            self.rgb_features, self.thermal_features
        )
        
        pose = self.fusion_module.estimate_pose(
            self.rgb_features, self.thermal_features, 
            (0.8, 0.2), matches
        )
        
        assert isinstance(pose, PoseWithUncertainty)
        assert len(pose.position) == 3
        assert len(pose.orientation) == 4  # quaternion
        assert 0.0 <= pose.confidence <= 1.0
        assert 0.0 <= pose.uncertainty <= 1.0
        assert pose.timestamp > 0
    
    def test_illumination_assessment(self):
        """Test RGB illumination assessment"""
        # Bright image
        bright_frame = np.full((480, 640), 200, dtype=np.uint8)
        bright_score = self.fusion_module.assess_illumination(bright_frame)
        
        # Dark image  
        dark_frame = np.full((480, 640), 50, dtype=np.uint8)
        dark_score = self.fusion_module.assess_illumination(dark_frame)
        
        # Bright should score higher
        assert bright_score > dark_score
        assert 0.0 <= bright_score <= 1.0
        assert 0.0 <= dark_score <= 1.0
    
    def test_thermal_contrast_assessment(self):
        """Test thermal contrast assessment"""
        # High contrast thermal image
        high_contrast = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        high_score = self.fusion_module.assess_thermal_contrast(high_contrast)
        
        # Low contrast thermal image
        low_contrast = np.full((480, 640), 128, dtype=np.uint8)
        low_score = self.fusion_module.assess_thermal_contrast(low_contrast)
        
        # High contrast should score higher
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
    
    def test_coordinate_transformation(self):
        """Test thermal-to-RGB coordinate transformation"""
        # Test with identity calibration matrix
        identity_matrix = np.eye(3)
        transformed = self.fusion_module._transform_coordinates(
            self.thermal_features, identity_matrix
        )
        
        assert len(transformed) == len(self.thermal_features)
        
        # With identity matrix, coordinates should be unchanged
        for orig, trans in zip(self.thermal_features, transformed):
            np.testing.assert_array_almost_equal(orig.pt, trans.pt, decimal=2)
    
    def test_similarity_computation(self):
        """Test feature similarity computation"""
        rgb_feat = self.rgb_features[0]
        thermal_feat = self.thermal_features[0]
        
        # Close features should have high similarity
        distance = np.linalg.norm(rgb_feat.pt - thermal_feat.pt)
        similarity = self.fusion_module._compute_similarity(
            rgb_feat, thermal_feat, distance
        )
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be reasonably high for close features
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        # Process some matches to generate stats
        for _ in range(5):
            self.fusion_module.match_cross_modal(
                self.rgb_features, self.thermal_features
            )
        
        stats = self.fusion_module.get_performance_stats()
        
        assert 'total_fusions' in stats
        assert 'average_matches_per_fusion' in stats
        assert 'average_processing_time_ms' in stats
        assert 'fusion_frequency_hz' in stats
        
        assert stats['total_fusions'] == 5
        assert stats['average_processing_time_ms'] >= 0
    
    def test_empty_features(self):
        """Test handling of empty feature lists"""
        empty_matches = self.fusion_module.match_cross_modal([], [])
        assert len(empty_matches) == 0
        
        # Should handle gracefully without crashing
        pose = self.fusion_module.estimate_pose([], [], (0.5, 0.5), [])
        assert pose.emergency_flag == True
        assert pose.confidence < 0.1


class TestFeature:
    """Test Feature dataclass"""
    
    def test_feature_creation(self):
        """Test Feature object creation"""
        pt = np.array([100.0, 200.0])
        descriptor = np.random.rand(256).astype(np.float32)
        
        feature = Feature(
            pt=pt,
            descriptor=descriptor,
            confidence=0.8,
            response=0.9
        )
        
        np.testing.assert_array_equal(feature.pt, pt)
        np.testing.assert_array_equal(feature.descriptor, descriptor)
        assert feature.confidence == 0.8
        assert feature.response == 0.9
    
    def test_feature_defaults(self):
        """Test Feature default values"""
        pt = np.array([50.0, 75.0])
        feature = Feature(pt=pt)
        
        assert feature.descriptor is None
        assert feature.confidence == 1.0
        assert feature.response == 0.0


class TestPoseWithUncertainty:
    """Test PoseWithUncertainty dataclass"""
    
    def test_pose_creation(self):
        """Test pose object creation"""
        position = np.array([1.0, 2.0, 3.0])
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        pose = PoseWithUncertainty(
            position=position,
            orientation=orientation,
            uncertainty=0.1,
            confidence=0.9,
            timestamp=12345.0
        )
        
        np.testing.assert_array_equal(pose.position, position)
        np.testing.assert_array_equal(pose.orientation, orientation)
        assert pose.uncertainty == 0.1
        assert pose.confidence == 0.9
        assert pose.timestamp == 12345.0
        assert pose.emergency_flag == False
    
    def test_emergency_pose(self):
        """Test emergency pose flag"""
        pose = PoseWithUncertainty(
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            uncertainty=1.0,
            confidence=0.01,
            timestamp=0.0,
            emergency_flag=True
        )
        
        assert pose.emergency_flag == True
        assert pose.confidence < 0.1
        assert pose.uncertainty == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
