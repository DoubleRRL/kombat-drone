"""
Thermal Target Detection using YOLOv8/v11
Fine-tuned for thermal signatures (vehicles, personnel, infrastructure)
"""
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class Detection:
    """Single target detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    thermal_signature: Optional[Dict[str, float]] = None


class ThermalPreprocessor:
    """Preprocessing pipeline for thermal images"""
    
    def __init__(self):
        self.normalization_method = "adaptive"  # "adaptive", "global", "histogram"
        self.noise_reduction = True
        self.contrast_enhancement = True
        
        # Thermal-specific parameters
        self.temp_range = (0, 255)  # Expected temperature range in image values
        self.noise_kernel_size = 3
        self.clahe_clip_limit = 2.0
    
    def normalize(self, thermal_frame: np.ndarray) -> np.ndarray:
        """
        Normalize thermal image for optimal YOLO performance
        
        Args:
            thermal_frame: Raw thermal image (typically 8-bit or 16-bit)
            
        Returns:
            Normalized thermal image ready for inference
        """
        # Convert to float32 for processing
        if thermal_frame.dtype != np.float32:
            thermal_frame = thermal_frame.astype(np.float32)
        
        # Apply noise reduction if enabled
        if self.noise_reduction:
            thermal_frame = self._reduce_noise(thermal_frame)
        
        # Normalize based on method
        if self.normalization_method == "adaptive":
            normalized = self._adaptive_normalize(thermal_frame)
        elif self.normalization_method == "global":
            normalized = self._global_normalize(thermal_frame)
        else:  # histogram
            normalized = self._histogram_normalize(thermal_frame)
        
        # Enhance contrast if enabled
        if self.contrast_enhancement:
            normalized = self._enhance_contrast(normalized)
        
        # Ensure proper range and type
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Convert to 3-channel if needed (YOLO expects RGB)
        if len(normalized.shape) == 2:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        
        return normalized
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction to thermal image"""
        # Bilateral filter preserves edges while reducing noise
        return cv2.bilateralFilter(
            image.astype(np.uint8), 
            self.noise_kernel_size, 
            50, 50
        ).astype(np.float32)
    
    def _adaptive_normalize(self, image: np.ndarray) -> np.ndarray:
        """Adaptive normalization based on local statistics"""
        # Use percentile-based normalization to handle outliers
        p_low, p_high = np.percentile(image, [2, 98])
        
        if p_high > p_low:
            normalized = (image - p_low) / (p_high - p_low) * 255
        else:
            normalized = image / np.max(image) * 255
        
        return normalized
    
    def _global_normalize(self, image: np.ndarray) -> np.ndarray:
        """Simple min-max normalization"""
        min_val, max_val = np.min(image), np.max(image)
        
        if max_val > min_val:
            normalized = (image - min_val) / (max_val - min_val) * 255
        else:
            normalized = image
        
        return normalized
    
    def _histogram_normalize(self, image: np.ndarray) -> np.ndarray:
        """Histogram equalization for better contrast"""
        # Convert to uint8 for histogram equalization
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(image_uint8)
        
        return equalized.astype(np.float32)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement"""
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=(8, 8)
        )
        
        # Apply CLAHE
        enhanced = clahe.apply(image_uint8)
        
        return enhanced.astype(np.float32)


class ThermalYOLO:
    """
    YOLOv8/v11 model fine-tuned for thermal target detection
    Handles vehicles, personnel, and infrastructure in thermal imagery
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "auto", half_precision: bool = False):
        """
        Initialize thermal YOLO detector
        
        Args:
            model_path: Path to trained model weights
            device: Device for inference ("auto", "cpu", "cuda", "mps")
            half_precision: Use FP16 inference for speed (GPU only)
        """
        self.model_path = model_path
        self.device = self._select_device(device)
        self.half_precision = half_precision and self.device != "cpu"
        
        # Load model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Enable half precision if requested and supported
            if self.half_precision:
                try:
                    self.model.half()
                    print(f"Loaded thermal YOLO model from {model_path} on {self.device} (FP16)")
                except:
                    self.half_precision = False
                    print(f"Loaded thermal YOLO model from {model_path} on {self.device} (FP16 not supported)")
            else:
                print(f"Loaded thermal YOLO model from {model_path} on {self.device}")
        except Exception as e:
            print(f"Failed to load model {model_path}, falling back to pretrained")
            self.model = YOLO("yolov8n.pt")  # Fallback to pretrained
            self.model.to(self.device)
        
        self.preprocessor = ThermalPreprocessor()
        
        # Detection parameters
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_detections = 1000
        
        # Thermal-specific class mapping (FLIR ADAS classes)
        self.thermal_classes = {
            0: "person",
            1: "bike", 
            2: "car",
            3: "motor",
            4: "bus",
            5: "train",
            6: "truck",
            7: "light"
        }
        
        # Performance tracking
        self.inference_times = []
        self.detection_stats = {
            'total_detections': 0,
            'avg_confidence': 0.0,
            'class_counts': {cls: 0 for cls in self.thermal_classes.values()}
        }
    
    def detect_targets(
        self, 
        thermal_frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        return_thermal_analysis: bool = True
    ) -> List[Detection]:
        """
        Detect targets in thermal image
        
        Args:
            thermal_frame: Input thermal image
            confidence_threshold: Override default confidence threshold
            return_thermal_analysis: Include thermal signature analysis
            
        Returns:
            List of Detection objects
        """
        start_time = time.time()
        
        # Preprocess thermal image (skip if it fails)
        try:
            processed_frame = self.preprocessor.normalize(thermal_frame)
        except Exception as e:
            print(f"Preprocessing failed, using original frame: {e}")
            processed_frame = thermal_frame
        
        # Set confidence threshold
        conf_thresh = confidence_threshold or self.confidence_threshold
        
        # Run inference
        try:
            results = self.model(
                processed_frame,
                conf=conf_thresh,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
        except Exception as e:
            print(f"YOLO inference failed: {e}")
            return []
        
        # Parse results
        detections = self._parse_results(
            results, thermal_frame, return_thermal_analysis
        )
        
        # Update performance stats
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self._update_detection_stats(detections)
        
        return detections
    
    def detect_targets_batch(
        self, 
        thermal_frames: List[np.ndarray],
        confidence_threshold: Optional[float] = None
    ) -> List[List[Detection]]:
        """
        Batch detection for multiple thermal frames
        More efficient for processing video sequences
        """
        if not thermal_frames:
            return []
        
        # Preprocess all frames
        processed_frames = [
            self.preprocessor.normalize(frame) for frame in thermal_frames
        ]
        
        # Batch inference
        conf_thresh = confidence_threshold or self.confidence_threshold
        
        try:
            results = self.model(
                processed_frames,
                conf=conf_thresh,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
        except Exception as e:
            print(f"Batch YOLO inference failed: {e}")
            return [[] for _ in thermal_frames]
        
        # Parse results for each frame
        batch_detections = []
        for i, (result, original_frame) in enumerate(zip(results, thermal_frames)):
            detections = self._parse_single_result(result, original_frame, True)
            batch_detections.append(detections)
            self._update_detection_stats(detections)
        
        return batch_detections
    
    def analyze_thermal_signature(
        self, 
        thermal_frame: np.ndarray, 
        bbox: Tuple[float, float, float, float]
    ) -> Dict[str, float]:
        """
        Analyze thermal signature within bounding box
        
        Args:
            thermal_frame: Original thermal image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary with thermal signature metrics
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Extract ROI
        roi = thermal_frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {"error": "invalid_bbox"}
        
        # Compute thermal statistics
        signature = {
            "mean_temp": float(np.mean(roi)),
            "max_temp": float(np.max(roi)),
            "min_temp": float(np.min(roi)),
            "temp_std": float(np.std(roi)),
            "temp_range": float(np.max(roi) - np.min(roi)),
            "hot_pixel_ratio": float(np.sum(roi > np.mean(roi) + np.std(roi)) / roi.size),
            "thermal_contrast": float(np.std(roi) / np.mean(roi)) if np.mean(roi) > 0 else 0.0
        }
        
        # Add contextual information
        full_mean = np.mean(thermal_frame)
        signature["relative_temperature"] = signature["mean_temp"] - full_mean
        signature["thermal_prominence"] = (
            signature["mean_temp"] / full_mean if full_mean > 0 else 1.0
        )
        
        return signature
    
    def _select_device(self, device: str) -> str:
        """Select optimal device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def _parse_results(
        self, 
        results, 
        original_frame: np.ndarray,
        include_thermal_analysis: bool
    ) -> List[Detection]:
        """Parse YOLO results into Detection objects"""
        if not results or len(results) == 0:
            return []
        
        return self._parse_single_result(results[0], original_frame, include_thermal_analysis)
    
    def _parse_single_result(
        self, 
        result, 
        original_frame: np.ndarray,
        include_thermal_analysis: bool
    ) -> List[Detection]:
        """Parse single YOLO result"""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract boxes, confidences, and classes
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Get class name
            class_name = self.thermal_classes.get(class_id, f"class_{class_id}")
            
            # Analyze thermal signature if requested
            thermal_signature = None
            if include_thermal_analysis:
                thermal_signature = self.analyze_thermal_signature(original_frame, box)
            
            detection = Detection(
                bbox=tuple(box),
                confidence=float(conf),
                class_id=class_id,
                class_name=class_name,
                thermal_signature=thermal_signature
            )
            
            detections.append(detection)
        
        return detections
    
    def _update_detection_stats(self, detections: List[Detection]):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(detections)
        
        if detections:
            # Update average confidence
            total_conf = sum(d.confidence for d in detections)
            self.detection_stats['avg_confidence'] = (
                (self.detection_stats['avg_confidence'] * 
                 (self.detection_stats['total_detections'] - len(detections)) +
                 total_conf) / self.detection_stats['total_detections']
            )
            
            # Update class counts
            for detection in detections:
                class_name = detection.class_name
                if class_name in self.detection_stats['class_counts']:
                    self.detection_stats['class_counts'][class_name] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics"""
        if self.inference_times:
            avg_inference_time = np.mean(self.inference_times[-100:])  # Last 100
            fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        else:
            avg_inference_time = 0
            fps = 0
        
        return {
            'total_detections': self.detection_stats['total_detections'],
            'average_confidence': self.detection_stats['avg_confidence'],
            'class_distribution': self.detection_stats['class_counts'].copy(),
            'average_inference_time_ms': avg_inference_time * 1000,
            'inference_fps': fps,
            'device': self.device,
            'model_path': self.model_path
        }
    
    def update_detection_parameters(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ):
        """Update detection parameters"""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold  
        if max_detections is not None:
            self.max_detections = max_detections
        
        print(f"Updated detection params: conf={self.confidence_threshold}, "
              f"iou={self.iou_threshold}, max_det={self.max_detections}")


def visualize_detections(
    image: np.ndarray, 
    detections: List[Detection],
    show_thermal_info: bool = True
) -> np.ndarray:
    """
    Visualize detections on image
    
    Args:
        image: Input image (RGB or thermal)
        detections: List of detections to visualize
        show_thermal_info: Show thermal signature information
        
    Returns:
        Image with visualized detections
    """
    vis_image = image.copy()
    
    # Convert to RGB if grayscale
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for all detections
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Background for label
        cv2.rectangle(
            vis_image, 
            (x1, y1 - label_size[1] - 10), 
            (x1 + label_size[0], y1), 
            color, -1
        )
        
        # Label text
        cv2.putText(
            vis_image, label, (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
        
        # Show thermal info if available
        if show_thermal_info and detection.thermal_signature:
            thermal_info = detection.thermal_signature
            temp_text = f"T:{thermal_info.get('mean_temp', 0):.1f}"
            cv2.putText(
                vis_image, temp_text, (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
            )
    
    return vis_image
