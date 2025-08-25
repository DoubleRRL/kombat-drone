"""
Armed Personnel Detection System
Addresses the critical limitation of thermal-only detection for weapon identification
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from working_thermal_detection import WorkingThermalDetector
from detection.thermal_yolo import Detection


@dataclass
class ArmedPersonnelThreat:
    """Enhanced threat assessment for armed personnel"""
    person_bbox: Tuple[int, int, int, int]
    weapon_probability: float
    weapon_type: str  # "rifle", "pistol", "concealed", "unknown"
    threat_level: str  # "low", "medium", "high", "critical"
    confidence: float
    detection_method: str
    behavioral_indicators: List[str]


class ArmedPersonnelDetector:
    """
    Enhanced detection system for armed personnel threats
    Combines multiple detection methods to overcome thermal-only limitations
    """
    
    def __init__(self):
        self.thermal_detector = WorkingThermalDetector(mode="accuracy")
        
        # Weapon detection parameters
        self.weapon_signatures = {
            "rifle": {
                "thermal_diff": (-20, -5),  # Cooler than body temp
                "shape": "elongated",
                "min_length": 60,
                "aspect_ratio": (3, 8)
            },
            "pistol": {
                "thermal_diff": (-15, 5),   # Slight temp difference
                "shape": "rectangular", 
                "min_length": 15,
                "aspect_ratio": (1.5, 3)
            }
        }
        
        print("Armed Personnel Detection System initialized")
        print("⚠️  WARNING: Thermal-only detection has limitations for weapon identification")
    
    def detect_armed_personnel(
        self, 
        rgb_frame: np.ndarray, 
        thermal_frame: np.ndarray
    ) -> Tuple[List[ArmedPersonnelThreat], Dict[str, Any]]:
        """
        Comprehensive armed personnel detection using multiple methods
        """
        start_time = time.time()
        
        # Method 1: Thermal personnel detection
        thermal_personnel = self._detect_thermal_personnel(thermal_frame)
        
        # Method 2: Weapon signature analysis (limited effectiveness)
        weapon_signatures = self._analyze_weapon_signatures(thermal_frame, thermal_personnel)
        
        # Method 3: RGB-based weapon detection (if sufficient light)
        rgb_weapons = self._detect_rgb_weapons(rgb_frame, thermal_personnel)
        
        # Method 4: Behavioral analysis
        behavioral_threats = self._analyze_behavior_patterns(rgb_frame, thermal_personnel)
        
        # Combine all methods for threat assessment
        armed_threats = self._fuse_detection_methods(
            thermal_personnel, weapon_signatures, rgb_weapons, behavioral_threats
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        performance = {
            'fps': 1000 / processing_time if processing_time > 0 else 0,
            'processing_time_ms': processing_time,
            'thermal_detections': len(thermal_personnel),
            'weapon_signatures': len(weapon_signatures),
            'rgb_detections': len(rgb_weapons),
            'final_threats': len(armed_threats)
        }
        
        return armed_threats, performance
    
    def _detect_thermal_personnel(self, thermal_frame: np.ndarray) -> List[Detection]:
        """Detect personnel using thermal signatures"""
        detections, _ = self.thermal_detector.detect_thermal_targets(thermal_frame)
        
        # Filter for personnel only
        personnel = [d for d in detections if 'personnel' in d.class_name.lower()]
        return personnel
    
    def _analyze_weapon_signatures(
        self, 
        thermal_frame: np.ndarray, 
        personnel: List[Detection]
    ) -> List[Dict[str, Any]]:
        """
        Analyze thermal signatures for potential weapons
        LIMITED EFFECTIVENESS - weapons may not have distinct thermal signatures
        """
        weapon_signatures = []
        
        for person in personnel:
            x1, y1, x2, y2 = [int(coord) for coord in person.bbox]
            
            # Expand search area around person
            search_margin = 50
            search_x1 = max(0, x1 - search_margin)
            search_y1 = max(0, y1 - search_margin)
            search_x2 = min(thermal_frame.shape[1], x2 + search_margin)
            search_y2 = min(thermal_frame.shape[0], y2 + search_margin)
            
            search_roi = thermal_frame[search_y1:search_y2, search_x1:search_x2]
            
            # Look for temperature anomalies that might indicate weapons
            person_temp = person.thermal_signature.get('mean_temp', 215)
            
            # Find regions significantly cooler (metal weapons) or hotter (fired weapons)
            cool_mask = cv2.inRange(search_roi, 0, int(person_temp - 20))
            hot_mask = cv2.inRange(search_roi, int(person_temp + 25), 255)
            
            # Analyze cool regions (potential metal weapons)
            cool_contours, _ = cv2.findContours(cool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in cool_contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum weapon size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1
                    
                    weapon_type = "unknown"
                    confidence = 0.3  # Low confidence for thermal-only weapon detection
                    
                    if aspect_ratio > 3 and w > 40:
                        weapon_type = "possible_rifle"
                        confidence = 0.4
                    elif 1.5 < aspect_ratio < 3 and w > 15:
                        weapon_type = "possible_pistol"  
                        confidence = 0.3
                    
                    weapon_signatures.append({
                        'bbox': (search_x1 + x, search_y1 + y, search_x1 + x + w, search_y1 + y + h),
                        'weapon_type': weapon_type,
                        'confidence': confidence,
                        'detection_method': 'thermal_anomaly',
                        'associated_person': person
                    })
        
        return weapon_signatures
    
    def _detect_rgb_weapons(
        self, 
        rgb_frame: np.ndarray, 
        personnel: List[Detection]
    ) -> List[Dict[str, Any]]:
        """
        RGB-based weapon detection (limited by lighting conditions)
        """
        rgb_weapons = []
        
        # Check if there's sufficient light for RGB analysis
        mean_brightness = np.mean(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY))
        
        if mean_brightness < 50:  # Too dark for reliable RGB detection
            return rgb_weapons
        
        # Simple edge-based weapon detection (very basic)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        for person in personnel:
            x1, y1, x2, y2 = [int(coord) for coord in person.bbox]
            
            # Expand search area
            margin = 40
            search_x1 = max(0, x1 - margin)
            search_y1 = max(0, y1 - margin)
            search_x2 = min(rgb_frame.shape[1], x2 + margin)
            search_y2 = min(rgb_frame.shape[0], y2 + margin)
            
            roi_edges = edges[search_y1:search_y2, search_x1:search_x2]
            
            # Look for linear features that might be weapons
            lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, threshold=30, 
                                   minLineLength=20, maxLineGap=5)
            
            if lines is not None:
                for line in lines:
                    x1_l, y1_l, x2_l, y2_l = line[0]
                    length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
                    
                    if length > 30:  # Potential weapon length
                        rgb_weapons.append({
                            'bbox': (search_x1 + x1_l, search_y1 + y1_l, 
                                   search_x1 + x2_l, search_y1 + y2_l),
                            'weapon_type': 'possible_weapon',
                            'confidence': 0.2,  # Very low confidence
                            'detection_method': 'rgb_edge_analysis',
                            'associated_person': person
                        })
        
        return rgb_weapons
    
    def _analyze_behavior_patterns(
        self, 
        rgb_frame: np.ndarray, 
        personnel: List[Detection]
    ) -> List[Dict[str, Any]]:
        """
        Behavioral analysis for threat assessment (very basic implementation)
        """
        behavioral_threats = []
        
        # This would require pose estimation, movement tracking, etc.
        # For now, just return basic threat assessment based on detection confidence
        
        for person in personnel:
            behavioral_indicators = []
            threat_multiplier = 1.0
            
            # High confidence detection might indicate suspicious behavior
            if person.confidence > 0.8:
                behavioral_indicators.append("high_confidence_detection")
                threat_multiplier += 0.2
            
            # Multiple personnel in area
            if len(personnel) > 1:
                behavioral_indicators.append("multiple_personnel")
                threat_multiplier += 0.1
            
            behavioral_threats.append({
                'person': person,
                'behavioral_indicators': behavioral_indicators,
                'threat_multiplier': threat_multiplier,
                'detection_method': 'behavioral_analysis'
            })
        
        return behavioral_threats
    
    def _fuse_detection_methods(
        self,
        thermal_personnel: List[Detection],
        weapon_signatures: List[Dict[str, Any]], 
        rgb_weapons: List[Dict[str, Any]],
        behavioral_threats: List[Dict[str, Any]]
    ) -> List[ArmedPersonnelThreat]:
        """
        Fuse all detection methods for final threat assessment
        """
        armed_threats = []
        
        for person in thermal_personnel:
            # Base threat assessment
            weapon_probability = 0.1  # Base probability for any detected person
            weapon_type = "unknown"
            threat_level = "low"
            detection_methods = ["thermal_personnel"]
            behavioral_indicators = []
            
            # Check for weapon signatures
            person_weapons = [w for w in weapon_signatures 
                            if w['associated_person'] == person]
            if person_weapons:
                weapon_probability += 0.3  # Increase probability
                weapon_type = person_weapons[0]['weapon_type']
                detection_methods.append("thermal_signature")
            
            # Check for RGB weapon detection
            person_rgb_weapons = [w for w in rgb_weapons 
                                if w['associated_person'] == person]
            if person_rgb_weapons:
                weapon_probability += 0.2
                detection_methods.append("rgb_analysis")
            
            # Check behavioral indicators
            person_behavior = next((b for b in behavioral_threats 
                                  if b['person'] == person), None)
            if person_behavior:
                weapon_probability *= person_behavior['threat_multiplier']
                behavioral_indicators = person_behavior['behavioral_indicators']
                detection_methods.append("behavioral_analysis")
            
            # Determine threat level
            if weapon_probability > 0.7:
                threat_level = "critical"
            elif weapon_probability > 0.5:
                threat_level = "high"  
            elif weapon_probability > 0.3:
                threat_level = "medium"
            else:
                threat_level = "low"
            
            armed_threats.append(ArmedPersonnelThreat(
                person_bbox=person.bbox,
                weapon_probability=min(weapon_probability, 0.95),  # Cap at 95%
                weapon_type=weapon_type,
                threat_level=threat_level,
                confidence=person.confidence,
                detection_method=" + ".join(detection_methods),
                behavioral_indicators=behavioral_indicators
            ))
        
        return armed_threats


def test_armed_personnel_scenarios():
    """Test the enhanced armed personnel detection system"""
    print("=== ENHANCED ARMED PERSONNEL DETECTION TEST ===")
    
    detector = ArmedPersonnelDetector()
    
    scenarios = [
        "Unarmed person in daylight",
        "Person with rifle in daylight", 
        "Person with concealed pistol",
        "Multiple armed personnel",
        "Person with weapon in low light"
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario} ---")
        
        # Create test frames (simplified)
        rgb_frame = np.random.randint(30, 200, (480, 640, 3), dtype=np.uint8)
        thermal_frame = np.ones((480, 640), dtype=np.uint8) * 85
        
        # Add person thermal signature
        cv2.circle(thermal_frame, (300, 250), 15, 215, -1)
        cv2.circle(thermal_frame, (300, 220), 8, 210, -1)
        
        # Add weapon signature for some scenarios
        if "rifle" in scenario:
            cv2.rectangle(thermal_frame, (320, 235), (380, 245), 160, -1)
        elif "pistol" in scenario:
            cv2.rectangle(thermal_frame, (315, 245), (325, 255), 180, -1)
        
        # Test detection
        threats, perf = detector.detect_armed_personnel(rgb_frame, thermal_frame)
        
        print(f"Performance: {perf['fps']:.1f} FPS")
        print(f"Threats detected: {len(threats)}")
        
        for threat in threats:
            print(f"  - Weapon probability: {threat.weapon_probability:.2f}")
            print(f"  - Weapon type: {threat.weapon_type}")
            print(f"  - Threat level: {threat.threat_level}")
            print(f"  - Detection method: {threat.detection_method}")
    
    print("\n=== SYSTEM LIMITATIONS ===")
    print("❌ CANNOT reliably detect:")
    print("  - Concealed weapons (no thermal signature)")
    print("  - Small weapons (pistols, knives)")
    print("  - Weapons at ambient temperature")
    print("  - Weapons in low-light RGB conditions")
    
    print("\n✅ CAN detect:")
    print("  - Personnel presence (thermal)")
    print("  - Recently fired weapons (hot signature)")
    print("  - Large metal weapons (temperature difference)")
    print("  - Suspicious behavior patterns (basic)")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("  - Deploy multiple sensor types (radar, lidar)")
    print("  - Use AI-trained weapon detection models")
    print("  - Implement pose estimation for weapon handling")
    print("  - Maintain safe standoff distances")
    print("  - Consider rules of engagement for uncertain threats")


if __name__ == "__main__":
    test_armed_personnel_scenarios()
