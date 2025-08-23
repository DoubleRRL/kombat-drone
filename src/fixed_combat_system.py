"""
Fixed Combat-Ready System with Working Thermal Detection
Replaces broken YOLO with temperature-based detection that actually works
"""
import numpy as np
import cv2
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import threading

from working_thermal_detection import WorkingThermalDetector
from combat_pipeline import CombatSLAM
from detection.thermal_yolo import Detection


@dataclass
class FixedCombatResult:
    """Fixed combat system result with working detection"""
    threats: List[Detection]
    position: Tuple[float, float, float]
    threat_level: str
    system_mode: str
    performance: Dict[str, float]
    detection_method: str


class FixedCombatSystem:
    """
    Fixed combat system using working thermal detection
    No more broken YOLO - uses temperature-based detection that actually finds threats
    """
    
    def __init__(self):
        print("Initializing FIXED Combat System with Working Detection...")
        
        # Use working thermal detector instead of broken YOLO
        self.thermal_detector = WorkingThermalDetector(mode="balanced")
        self.slam = CombatSLAM()
        
        # System state
        self.current_mode = "normal"
        self.threat_history = []
        self.frame_count = 0
        
        # Performance tracking
        self.performance_stats = []
        
        print("✅ Fixed combat system ready - thermal detection WORKING")
    
    def process_combat_frame(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        force_mode: Optional[str] = None
    ) -> FixedCombatResult:
        """
        Process frame with working thermal detection
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Determine processing mode based on threat history
        processing_mode = force_mode or self._select_processing_mode()
        
        # Run thermal detection (actually works!)
        threats, detection_perf = self.thermal_detector.detect_thermal_targets(thermal_frame)
        
        # Run SLAM
        position = self.slam.update_position(rgb_frame)
        
        # Assess threat level
        threat_level = self._assess_threat_level(threats)
        
        # Update system state
        total_time = (time.time() - start_time) * 1000
        self.threat_history.append(threats)
        if len(self.threat_history) > 20:
            self.threat_history = self.threat_history[-20:]
        
        self.performance_stats.append(total_time)
        if len(self.performance_stats) > 100:
            self.performance_stats = self.performance_stats[-100:]
        
        return FixedCombatResult(
            threats=threats,
            position=position,
            threat_level=threat_level,
            system_mode=processing_mode,
            performance={
                'fps': 1000 / total_time if total_time > 0 else 0,
                'total_time_ms': total_time,
                'detection_fps': detection_perf['fps'],
                'detection_count': len(threats),
                'avg_confidence': detection_perf['avg_confidence']
            },
            detection_method="thermal_temperature_analysis"
        )
    
    def _select_processing_mode(self) -> str:
        """Select processing mode based on threat history"""
        if not self.threat_history:
            return "normal"
        
        recent_threats = self.threat_history[-5:]
        avg_threats = np.mean([len(threats) for threats in recent_threats])
        high_conf_threats = np.mean([
            len([t for t in threats if t.confidence > 0.7])
            for threats in recent_threats
        ])
        
        if high_conf_threats > 2:
            return "high_threat"
        elif avg_threats < 1:
            return "surveillance"
        else:
            return "normal"
    
    def _assess_threat_level(self, threats: List[Detection]) -> str:
        """Assess threat level based on detections"""
        if not threats:
            return "clear"
        
        # Count high-confidence threats
        high_conf = [t for t in threats if t.confidence > 0.7]
        vehicles = [t for t in threats if "vehicle" in t.class_name]
        personnel = [t for t in threats if "personnel" in t.class_name]
        aircraft = [t for t in threats if "aircraft" in t.class_name]
        
        if aircraft or len(high_conf) > 3:
            return "critical"
        elif vehicles and personnel:
            return "high"
        elif len(threats) > 2:
            return "elevated"
        else:
            return "low"
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        if not self.performance_stats:
            return {'avg_fps': 0, 'avg_latency': 0}
        
        times = np.array(self.performance_stats)
        return {
            'avg_fps': 1000 / np.mean(times),
            'min_fps': 1000 / np.max(times),
            'max_fps': 1000 / np.min(times),
            'avg_latency_ms': np.mean(times),
            'p95_latency_ms': np.percentile(times, 95)
        }


def benchmark_fixed_system():
    """Benchmark the fixed combat system"""
    print("=== FIXED COMBAT SYSTEM BENCHMARK ===")
    print("Using WORKING thermal detection instead of broken YOLO\n")
    
    system = FixedCombatSystem()
    
    # Create realistic test scenario
    rgb_frame = np.random.randint(30, 70, (480, 640, 3), dtype=np.uint8)  # Dark
    thermal_frame = np.ones((480, 640), dtype=np.uint8) * 85  # Background
    
    # Add multiple realistic thermal signatures
    # Moving convoy
    for i in range(3):
        x = 100 + i * 120
        y = 200 + i * 20
        # Vehicle body
        cv2.rectangle(thermal_frame, (x, y), (x+80, y+40), 190 + i*10, -1)
        # Hot engine
        cv2.rectangle(thermal_frame, (x+10, y+5), (x+30, y+15), 240, -1)
    
    # Personnel patrol
    for i in range(2):
        x = 400 + i * 60
        y = 300 + i * 30
        cv2.circle(thermal_frame, (x, y), 15, 215, -1)  # Body heat
        cv2.circle(thermal_frame, (x, y-20), 6, 210, -1)  # Head
    
    # Aircraft
    cv2.ellipse(thermal_frame, (300, 100), (50, 20), 0, 0, 360, 200, -1)
    cv2.circle(thermal_frame, (280, 100), 8, 250, -1)  # Engine
    cv2.circle(thermal_frame, (320, 100), 8, 250, -1)  # Engine
    
    print("Test scenario: Convoy (3 vehicles), Patrol (2 personnel), Aircraft (1)")
    
    # Run benchmark
    results = []
    threat_counts = []
    
    print("\nProcessing 50 frames...")
    for i in range(50):
        # Add some movement/variation
        offset_x = int(5 * np.sin(i * 0.2))
        offset_y = int(3 * np.cos(i * 0.3))
        
        # Shift thermal signatures slightly
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        thermal_test = cv2.warpAffine(thermal_frame, M, (thermal_frame.shape[1], thermal_frame.shape[0]))
        
        result = system.process_combat_frame(rgb_frame, thermal_test)
        results.append(result)
        threat_counts.append(len(result.threats))
        
        if i % 10 == 0:
            print(f"  Frame {i}: {result.performance['fps']:.1f} FPS, "
                  f"{len(result.threats)} threats, Level: {result.threat_level}")
    
    # Analyze results
    avg_fps = np.mean([r.performance['fps'] for r in results])
    avg_threats = np.mean(threat_counts)
    threat_levels = [r.threat_level for r in results]
    
    perf_summary = system.get_performance_summary()
    
    print(f"\n=== BENCHMARK RESULTS ===")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average Threats Detected: {avg_threats:.1f}")
    print(f"Threat Levels: {dict(zip(*np.unique(threat_levels, return_counts=True)))}")
    print(f"P95 Latency: {perf_summary['p95_latency_ms']:.1f}ms")
    
    # Success criteria
    success_criteria = {
        'fps_target': avg_fps >= 30,
        'detection_target': avg_threats >= 4,  # Should detect most of the 6 targets
        'threat_assessment': 'high' in threat_levels or 'critical' in threat_levels,
        'system_stability': len([r for r in results if len(r.threats) > 0]) >= 40  # 80% detection rate
    }
    
    print(f"\n=== SUCCESS CRITERIA ===")
    for criterion, met in success_criteria.items():
        status = "✅" if met else "❌"
        print(f"{criterion.replace('_', ' ').title()}: {status}")
    
    overall_success = all(success_criteria.values())
    print(f"\nOVERALL SYSTEM STATUS: {'✅ COMBAT READY' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    
    return results, perf_summary, success_criteria


if __name__ == "__main__":
    results, perf, criteria = benchmark_fixed_system()
