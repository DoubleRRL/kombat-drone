"""
Tests for thermal detection module
Validates YOLO detection and thermal preprocessing
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from detection.thermal_yolo import ThermalYOLO, ThermalPreprocessor, Detection


class TestThermalPreprocessor:
    """Test thermal image preprocessing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = ThermalPreprocessor()
    
    def test_normalize_8bit_image(self):
        """Test normalization of 8-bit thermal image"""
        # Create test thermal image
        thermal_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        normalized = self.preprocessor.normalize(thermal_image)
        
        # Check output properties
        assert normalized.dtype == np.uint8
        assert normalized.shape == (480, 640, 3)  # Should be RGB
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 255
    
    def test_normalize_16bit_image(self):
        """Test normalization of 16-bit thermal image"""
        # Create test 16-bit thermal image
        thermal_image = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
        
        normalized = self.preprocessor.normalize(thermal_image)
        
        # Check output properties
        assert normalized.dtype == np.uint8
        assert normalized.shape == (480, 640, 3)
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 255
    
    def test_adaptive_normalization(self):
        """Test adaptive normalization method"""
        self.preprocessor.normalization_method = "adaptive"
        
        # Create image with outliers
        thermal_image = np.full((100, 100), 128, dtype=np.uint8)
        thermal_image[0, 0] = 255  # Hot spot
        thermal_image[99, 99] = 0  # Cold spot
        
        normalized = self.preprocessor.normalize(thermal_image)
        
        # Should handle outliers well
        assert normalized.dtype == np.uint8
        assert normalized.shape == (100, 100, 3)
    
    def test_global_normalization(self):
        """Test global min-max normalization"""
        self.preprocessor.normalization_method = "global"
        
        thermal_image = np.linspace(0, 255, 100*100).reshape(100, 100).astype(np.uint8)
        normalized = self.preprocessor.normalize(thermal_image)
        
        assert normalized.dtype == np.uint8
        assert normalized.shape == (100, 100, 3)
    
    def test_noise_reduction(self):
        """Test noise reduction functionality"""
        self.preprocessor.noise_reduction = True
        
        # Create noisy image
        clean_image = np.full((100, 100), 128, dtype=np.uint8)
        noisy_image = clean_image + np.random.normal(0, 20, (100, 100)).astype(np.int16)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        # Process with and without noise reduction
        self.preprocessor.noise_reduction = True
        denoised = self.preprocessor.normalize(noisy_image)
        
        self.preprocessor.noise_reduction = False
        not_denoised = self.preprocessor.normalize(noisy_image)
        
        # Denoised should be smoother (this is a simple check)
        assert denoised.dtype == np.uint8
        assert not_denoised.dtype == np.uint8
    
    def test_contrast_enhancement(self):
        """Test contrast enhancement"""
        self.preprocessor.contrast_enhancement = True
        
        # Low contrast image
        low_contrast = np.random.normal(128, 10, (100, 100)).astype(np.uint8)
        
        enhanced = self.preprocessor.normalize(low_contrast)
        
        assert enhanced.dtype == np.uint8
        assert enhanced.shape == (100, 100, 3)


class TestThermalYOLO:
    """Test thermal YOLO detection"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Use CPU for testing to avoid GPU dependencies
        self.detector = ThermalYOLO(device="cpu")
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        assert self.detector.device == "cpu"
        assert self.detector.model is not None
        assert self.detector.preprocessor is not None
        assert self.detector.confidence_threshold > 0
        assert self.detector.iou_threshold > 0
    
    @patch('ultralytics.YOLO')
    def test_detect_targets_mock(self, mock_yolo):
        """Test target detection with mocked YOLO"""
        # Mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        
        # Mock detection results
        mock_boxes = Mock()
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([[100, 100, 200, 200]])
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.8])
        mock_boxes.cls.cpu().numpy.return_value = np.array([0])  # person class
        mock_boxes.__len__ = Mock(return_value=1)  # Mock len() for boxes
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        self.detector.model = mock_model
        
        # Test detection
        thermal_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        detections = self.detector.detect_targets(thermal_frame)
        
        assert len(detections) == 1
        detection = detections[0]
        assert isinstance(detection, Detection)
        assert detection.confidence == 0.8
        assert detection.class_name == "person"
        assert len(detection.bbox) == 4
    
    def test_thermal_signature_analysis(self):
        """Test thermal signature analysis"""
        # Create thermal image with hot spot
        thermal_frame = np.full((480, 640), 100, dtype=np.uint8)
        thermal_frame[200:300, 300:400] = 200  # Hot region
        
        bbox = (300, 200, 400, 300)  # x1, y1, x2, y2
        signature = self.detector.analyze_thermal_signature(thermal_frame, bbox)
        
        assert isinstance(signature, dict)
        assert 'mean_temp' in signature
        assert 'max_temp' in signature
        assert 'min_temp' in signature
        assert 'temp_std' in signature
        assert 'thermal_contrast' in signature
        
        # Hot region should have higher mean temperature
        assert signature['mean_temp'] > 150
    
    def test_batch_detection(self):
        """Test batch detection functionality"""
        # Create multiple thermal frames
        frames = []
        for i in range(3):
            frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
            frames.append(frame)
        
        # Mock the model to avoid actual inference
        with patch.object(self.detector, 'model') as mock_model:
            # Create mock results for each frame
            mock_results = []
            for _ in frames:
                mock_result = Mock()
                mock_result.boxes = None  # No detections
                mock_results.append(mock_result)
            
            mock_model.return_value = mock_results
            
            batch_results = self.detector.detect_targets_batch(frames)
            
            assert len(batch_results) == 3
            for result in batch_results:
                assert isinstance(result, list)
    
    def test_parameter_updates(self):
        """Test detection parameter updates"""
        original_conf = self.detector.confidence_threshold
        original_iou = self.detector.iou_threshold
        
        self.detector.update_detection_parameters(
            confidence_threshold=0.5,
            iou_threshold=0.3,
            max_detections=500
        )
        
        assert self.detector.confidence_threshold == 0.5
        assert self.detector.iou_threshold == 0.3
        assert self.detector.max_detections == 500
        
        # Test partial updates
        self.detector.update_detection_parameters(confidence_threshold=0.7)
        assert self.detector.confidence_threshold == 0.7
        assert self.detector.iou_threshold == 0.3  # Should remain unchanged
    
    def test_performance_stats(self):
        """Test performance statistics"""
        # Initial stats should be zero
        stats = self.detector.get_performance_stats()
        
        assert stats['total_detections'] == 0
        assert stats['average_confidence'] == 0.0
        assert 'device' in stats
        assert 'model_path' in stats
        
        # Mock some detections to update stats
        mock_detections = [
            Detection(bbox=(0, 0, 100, 100), confidence=0.8, class_id=0, class_name="person"),
            Detection(bbox=(100, 100, 200, 200), confidence=0.6, class_id=2, class_name="car")
        ]
        
        self.detector._update_detection_stats(mock_detections)
        
        updated_stats = self.detector.get_performance_stats()
        assert updated_stats['total_detections'] == 2
        assert updated_stats['average_confidence'] == 0.7  # (0.8 + 0.6) / 2
    
    def test_device_selection(self):
        """Test device selection logic"""
        # Test auto selection
        device = self.detector._select_device("auto")
        assert device in ["cpu", "cuda", "mps"]
        
        # Test explicit selection
        assert self.detector._select_device("cpu") == "cpu"
        assert self.detector._select_device("cuda") == "cuda"
        assert self.detector._select_device("mps") == "mps"


class TestDetection:
    """Test Detection dataclass"""
    
    def test_detection_creation(self):
        """Test Detection object creation"""
        bbox = (100.0, 150.0, 200.0, 250.0)
        thermal_sig = {"mean_temp": 180.5, "max_temp": 220.0}
        
        detection = Detection(
            bbox=bbox,
            confidence=0.85,
            class_id=2,
            class_name="car",
            thermal_signature=thermal_sig
        )
        
        assert detection.bbox == bbox
        assert detection.confidence == 0.85
        assert detection.class_id == 2
        assert detection.class_name == "car"
        assert detection.thermal_signature == thermal_sig
    
    def test_detection_without_thermal_signature(self):
        """Test Detection without thermal signature"""
        detection = Detection(
            bbox=(0, 0, 50, 50),
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        
        assert detection.thermal_signature is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
