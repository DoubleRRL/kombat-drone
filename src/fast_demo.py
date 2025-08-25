"""
Fast demo mode with aggressive optimizations for README demonstration
Optimized for speed over accuracy
"""
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class FastDetection:
    """Lightweight detection result"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str


@dataclass 
class FastPose:
    """Lightweight pose result"""
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    confidence: float


class FastThermalDetector:
    """Ultra-fast thermal detection using OpenCV instead of YOLO"""
    
    def __init__(self):
        # Simple blob detector for thermal signatures
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 5000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        self.detector = cv2.SimpleBlobDetector_create(params)
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=50
        )
        
    def detect(self, thermal_frame: np.ndarray) -> List[FastDetection]:
        """Ultra-fast detection using OpenCV"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(thermal_frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(area / 1000.0, 1.0)  # Simple confidence based on area
                
                # Simple classification based on aspect ratio
                aspect_ratio = w / h if h > 0 else 1.0
                if aspect_ratio > 1.5:
                    class_name = "vehicle"
                elif 0.5 < aspect_ratio < 1.5:
                    class_name = "person"
                else:
                    class_name = "object"
                
                detections.append(FastDetection(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidence,
                    class_name=class_name
                ))
        
        return detections[:10]  # Limit to 10 detections for speed


class FastSLAM:
    """Ultra-fast SLAM using optical flow"""
    
    def __init__(self):
        self.prev_frame = None
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=50,  # Reduced for speed
            qualityLevel=0.1,
            minDistance=10,
            blockSize=7
        )
        
        self.tracks = []
        
    def process(self, rgb_frame: np.ndarray) -> FastPose:
        """Ultra-fast SLAM using optical flow"""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            # Initialize tracking
            self.prev_frame = gray
            corners = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            if corners is not None:
                self.tracks = corners.reshape(-1, 1, 2)
            return FastPose((0, 0, 0), (0, 0, 0), 1.0)
        
        # Track features
        if len(self.tracks) > 0:
            new_tracks, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, self.tracks, None, **self.lk_params
            )
            
            # Keep only good tracks
            status_flat = status.flatten()
            good_tracks = new_tracks[status_flat == 1]
            old_tracks = self.tracks[status_flat == 1]
            
            if len(good_tracks) > 5:
                # Estimate motion from optical flow
                flow = good_tracks - old_tracks
                avg_flow = np.mean(flow.reshape(-1, 2), axis=0)
                
                # Simple motion estimation (very basic)
                self.position[0] += float(avg_flow[0]) * 0.01  # Scale factor
                self.position[1] += float(avg_flow[1]) * 0.01
                self.position[2] += 0.001  # Assume forward motion
                
                confidence = min(len(good_tracks) / 50.0, 1.0)
                
                self.tracks = good_tracks
            else:
                confidence = 0.1
                
            # Add new features if needed
            if len(self.tracks) < 20:
                corners = cv2.goodFeaturesToTrack(gray, **self.feature_params)
                if corners is not None:
                    self.tracks = np.vstack([self.tracks, corners.reshape(-1, 1, 2)])
        else:
            confidence = 0.0
        
        self.prev_frame = gray
        
        return FastPose(
            position=tuple(self.position),
            rotation=tuple(self.rotation),
            confidence=confidence
        )


class FastPipeline:
    """Ultra-fast pipeline for demonstration"""
    
    def __init__(self):
        self.detector = FastThermalDetector()
        self.slam = FastSLAM()
        self.frame_count = 0
        
    def process(
        self, 
        rgb_frame: np.ndarray, 
        thermal_frame: np.ndarray
    ) -> Dict[str, Any]:
        """Process frames with maximum speed"""
        self.frame_count += 1
        
        # Resize frames for speed (smaller = faster)
        rgb_small = cv2.resize(rgb_frame, (320, 240))
        thermal_small = cv2.resize(thermal_frame, (320, 240))
        
        # Run detection and SLAM
        detections = self.detector.detect(thermal_small)
        pose = self.slam.process(rgb_small)
        
        # Scale detection bounding boxes back up
        scale_x = rgb_frame.shape[1] / 320
        scale_y = rgb_frame.shape[0] / 240
        
        scaled_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            scaled_bbox = (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            )
            scaled_detections.append(FastDetection(
                bbox=scaled_bbox,
                confidence=det.confidence,
                class_name=det.class_name
            ))
        
        return {
            'pose': pose,
            'detections': scaled_detections,
            'system_health': {'overall_health': 'healthy'},
            'frame_count': self.frame_count
        }


def benchmark_fast_pipeline():
    """Benchmark the fast pipeline"""
    print("=== FAST PIPELINE BENCHMARK ===")
    
    pipeline = FastPipeline()
    
    # Generate test frames
    rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    thermal_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Add some synthetic objects for detection
    cv2.rectangle(rgb_frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(thermal_frame, (100, 100), (200, 200), 255, -1)
    
    # Warm-up
    for _ in range(5):
        pipeline.process(rgb_frame, thermal_frame)
    
    # Benchmark
    times = []
    for i in range(50):
        # Add some variation
        rgb_test = rgb_frame.copy()
        thermal_test = thermal_frame.copy()
        
        # Move the object slightly
        offset_x = int(10 * np.sin(i * 0.1))
        offset_y = int(5 * np.cos(i * 0.1))
        
        rgb_test = np.roll(rgb_test, offset_x, axis=1)
        thermal_test = np.roll(thermal_test, offset_x, axis=1)
        
        start = time.time()
        result = pipeline.process(rgb_test, thermal_test)
        end = time.time()
        
        times.append((end - start) * 1000)
        
        if i % 10 == 0:
            print(f"Frame {i}: {times[-1]:.1f}ms, Detections: {len(result['detections'])}, "
                  f"SLAM Conf: {result['pose'].confidence:.2f}")
    
    avg_time = np.mean(times)
    avg_fps = 1000 / avg_time
    min_time = np.min(times)
    max_fps = 1000 / min_time
    
    print(f"\\nFAST PIPELINE RESULTS:")
    print(f"Average: {avg_time:.1f}ms ({avg_fps:.1f} FPS)")
    print(f"Best: {min_time:.1f}ms ({max_fps:.1f} FPS)")
    print(f"Worst: {np.max(times):.1f}ms ({1000/np.max(times):.1f} FPS)")
    print(f"Std: {np.std(times):.1f}ms")
    
    return avg_fps, max_fps


if __name__ == "__main__":
    avg_fps, max_fps = benchmark_fast_pipeline()
