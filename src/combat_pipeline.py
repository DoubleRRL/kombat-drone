"""
COMBAT PIPELINE - Maximum Speed Priority
Strips everything non-essential, optimized for 100+ FPS
"""
import numpy as np
import cv2
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThreatDetection:
    """Minimal threat detection result"""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    threat_type: str


class CombatThermalDetector:
    """
    Ultra-fast threat detection using pure OpenCV
    NO YOLO - too slow for combat
    """
    
    def __init__(self):
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=30,
            history=20  # Reduced history for speed
        )
        
        # Thermal signature templates (pre-computed)
        self.vehicle_template = np.ones((40, 80), dtype=np.uint8) * 200
        self.person_template = np.ones((60, 30), dtype=np.uint8) * 220
        
        # Optimized contour parameters
        self.min_area = 200
        self.max_area = 10000
        
    def detect_threats(self, thermal_frame: np.ndarray) -> List[ThreatDetection]:
        """Ultra-fast threat detection - target <5ms"""
        start_time = time.time()
        
        # Resize for speed (smaller = faster)
        small_frame = cv2.resize(thermal_frame, (160, 120))  # Very small
        
        # Background subtraction for motion
        fg_mask = self.bg_subtractor.apply(small_frame)
        
        # Morphological operations (minimal)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        scale_x = thermal_frame.shape[1] / 160
        scale_y = thermal_frame.shape[0] / 120
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale back to original size
                x = int(x * scale_x)
                y = int(y * scale_y) 
                w = int(w * scale_x)
                h = int(h * scale_y)
                
                # Simple classification based on aspect ratio
                aspect_ratio = w / h if h > 0 else 1.0
                if aspect_ratio > 2.0:
                    threat_type = "vehicle"
                    confidence = 0.8
                elif 0.3 < aspect_ratio < 1.5:
                    threat_type = "personnel"
                    confidence = 0.7
                else:
                    threat_type = "unknown"
                    confidence = 0.5
                
                detections.append(ThreatDetection(
                    x=x, y=y, w=w, h=h,
                    confidence=confidence,
                    threat_type=threat_type
                ))
        
        # Limit detections for speed
        return detections[:5]  # Max 5 detections


class CombatSLAM:
    """
    Minimal SLAM for combat - just basic position tracking
    """
    
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.prev_frame = None
        
        # Optical flow parameters (minimal)
        self.lk_params = dict(
            winSize=(10, 10),  # Smaller window
            maxLevel=1,        # Fewer pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.1)
        )
        
        # Feature parameters (minimal)
        self.feature_params = dict(
            maxCorners=20,     # Very few features
            qualityLevel=0.2,  # Lower quality for speed
            minDistance=20,
            blockSize=5
        )
        
        self.tracks = []
        
    def update_position(self, rgb_frame: np.ndarray) -> Tuple[float, float, float]:
        """Minimal position update"""
        # Resize for speed
        small_frame = cv2.resize(rgb_frame, (160, 120))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            # Initialize tracking points
            corners = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            if corners is not None:
                self.tracks = corners
            return tuple(self.position)
        
        # Track features if we have them
        if len(self.tracks) > 0:
            new_tracks, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, self.tracks, None, **self.lk_params
            )
            
            # Keep good tracks
            good_tracks = new_tracks[status.flatten() == 1]
            
            if len(good_tracks) > 5:
                # Simple motion estimation
                old_good = self.tracks[status.flatten() == 1]
                flow = good_tracks - old_good
                avg_flow = np.mean(flow.reshape(-1, 2), axis=0)
                
                # Update position (very basic)
                self.position[0] += float(avg_flow[0]) * 0.01
                self.position[1] += float(avg_flow[1]) * 0.01
                self.position[2] += 0.005  # Assume forward motion
                
                self.tracks = good_tracks
            
            # Refresh tracks if too few
            if len(self.tracks) < 10:
                corners = cv2.goodFeaturesToTrack(gray, **self.feature_params)
                if corners is not None:
                    if len(self.tracks) > 0:
                        self.tracks = np.vstack([self.tracks, corners])
                    else:
                        self.tracks = corners
        
        self.prev_frame = gray
        return tuple(self.position)


class CombatPipeline:
    """
    COMBAT PIPELINE - Absolute minimum processing for maximum speed
    Target: 100+ FPS sustained
    """
    
    def __init__(self):
        print("Initializing COMBAT Pipeline - Maximum Speed Priority")
        
        self.detector = CombatThermalDetector()
        self.slam = CombatSLAM()
        
        self.frame_count = 0
        self.performance_times = []
        
        print("Combat pipeline ready - targeting 100+ FPS")
    
    def process_combat_frame(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Combat frame processing - MAXIMUM SPEED
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Process detection and SLAM in parallel (but keep it simple)
        import threading
        
        detection_result = [None]
        slam_result = [None]
        
        def detect():
            detection_result[0] = self.detector.detect_threats(thermal_frame)
        
        def track():
            slam_result[0] = self.slam.update_position(rgb_frame)
        
        # Run in parallel
        det_thread = threading.Thread(target=detect)
        slam_thread = threading.Thread(target=track)
        
        det_thread.start()
        slam_thread.start()
        
        det_thread.join()
        slam_thread.join()
        
        detections = detection_result[0] or []
        position = slam_result[0] or (0, 0, 0)
        
        # Minimal threat assessment
        threat_count = len(detections)
        if threat_count == 0:
            threat_level = "clear"
        elif threat_count < 3:
            threat_level = "low"
        else:
            threat_level = "high"
        
        # Performance tracking
        total_time = (time.time() - start_time) * 1000
        self.performance_times.append(total_time)
        
        # Keep only recent data
        if len(self.performance_times) > 50:
            self.performance_times = self.performance_times[-50:]
        
        return {
            'threats': detections,
            'position': position,
            'threat_level': threat_level,
            'frame_count': self.frame_count,
            'fps': 1000 / total_time if total_time > 0 else 0,
            'latency_ms': total_time
        }
    
    def get_combat_stats(self) -> Dict[str, float]:
        """Get combat performance statistics"""
        if not self.performance_times:
            return {}
        
        times = np.array(self.performance_times)
        return {
            'avg_fps': 1000 / np.mean(times),
            'min_fps': 1000 / np.max(times),
            'max_fps': 1000 / np.min(times),
            'avg_latency': np.mean(times),
            'max_latency': np.max(times)
        }


def combat_benchmark():
    """Combat pipeline benchmark - targeting 100+ FPS"""
    print("=== COMBAT PIPELINE BENCHMARK ===")
    print("MAXIMUM SPEED PRIORITY - Targeting 100+ FPS")
    
    pipeline = CombatPipeline()
    
    # Test data with threats
    rgb_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)  # Smaller
    thermal_frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    
    # Add threats
    cv2.rectangle(thermal_frame, (50, 50), (100, 90), 255, -1)  # Vehicle
    cv2.circle(thermal_frame, (200, 150), 15, 220, -1)  # Personnel
    
    print("\\nRunning combat performance test...")
    
    # Warm-up
    for _ in range(10):
        pipeline.process_combat_frame(rgb_frame, thermal_frame)
    
    # Combat test
    fps_results = []
    
    for i in range(100):  # More frames for combat test
        frame_start = time.time()
        
        result = pipeline.process_combat_frame(rgb_frame, thermal_frame)
        
        frame_time = time.time() - frame_start
        fps = 1 / frame_time if frame_time > 0 else 0
        fps_results.append(fps)
        
        if i % 20 == 0:
            print(f"  Frame {i:3d}: {fps:6.1f} FPS, "
                  f"Threats: {len(result['threats'])}, "
                  f"Level: {result['threat_level']}")
    
    # Final stats
    stats = pipeline.get_combat_stats()
    avg_fps = np.mean(fps_results)
    min_fps = np.min(fps_results)
    max_fps = np.max(fps_results)
    
    print(f"\\n=== COMBAT RESULTS ===")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Min FPS: {min_fps:.1f}")
    print(f"Max FPS: {max_fps:.1f}")
    print(f"Average Latency: {stats.get('avg_latency', 0):.1f}ms")
    print(f"Max Latency: {stats.get('max_latency', 0):.1f}ms")
    
    if avg_fps >= 100:
        print("🎯 COMBAT READY: Exceeds 100 FPS target!")
    elif avg_fps >= 60:
        print("✅ ACCEPTABLE: Above 60 FPS minimum")
    else:
        print("❌ INSUFFICIENT: Below combat requirements")
    
    return avg_fps


if __name__ == "__main__":
    combat_fps = combat_benchmark()
