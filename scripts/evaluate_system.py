"""
System evaluation script
Tests the complete thermal SLAM + detection pipeline on standard datasets
"""
import numpy as np
import cv2
import time
import json
from pathlib import Path
import argparse
from typing import Dict, List, Any
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import MultiModalPipeline
from detection.thermal_yolo import visualize_detections


def evaluate_on_tum_dataset(
    dataset_path: str, 
    pipeline: MultiModalPipeline,
    max_frames: int = 100
) -> Dict[str, Any]:
    """
    Evaluate pipeline on TUM RGB-D dataset
    
    Args:
        dataset_path: Path to TUM dataset
        pipeline: Initialized pipeline
        max_frames: Maximum frames to process
        
    Returns:
        Evaluation metrics
    """
    dataset_dir = Path(dataset_path)
    rgb_dir = dataset_dir / "rgb"
    
    if not rgb_dir.exists():
        raise ValueError(f"RGB directory not found: {rgb_dir}")
    
    # Get RGB images
    rgb_files = sorted(list(rgb_dir.glob("*.png")))[:max_frames]
    
    # Load ground truth poses if available
    groundtruth_file = dataset_dir / "groundtruth.txt"
    gt_poses = {}
    if groundtruth_file.exists():
        with open(groundtruth_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    position = [float(parts[1]), float(parts[2]), float(parts[3])]
                    gt_poses[timestamp] = position
    
    # Evaluation metrics
    results = {
        'total_frames': 0,
        'successful_tracks': 0,
        'processing_times': [],
        'slam_confidences': [],
        'detection_counts': [],
        'pose_errors': [],
        'system_health_scores': []
    }
    
    print(f"Evaluating on {len(rgb_files)} frames from {dataset_dir.name}")
    
    for i, rgb_file in enumerate(rgb_files):
        start_time = time.time()
        
        # Load RGB image
        rgb_frame = cv2.imread(str(rgb_file))
        if rgb_frame is None:
            continue
        
        # Create mock thermal frame (in practice would load actual thermal)
        thermal_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        # Mock IMU data
        imu_data = {
            'accel': [0.0, 0.0, 9.8],
            'gyro': [0.0, 0.0, 0.0]
        }
        
        # Process frame
        try:
            result = pipeline.process_streams(rgb_frame, thermal_frame, imu_data)
            
            # Extract metrics
            pose = result['pose']
            detections = result['detections']
            performance = result['performance']
            health = result['system_health']
            
            results['total_frames'] += 1
            results['processing_times'].append(performance['total_time_ms'])
            results['slam_confidences'].append(pose.confidence)
            results['detection_counts'].append(len(detections))
            
            if pose.confidence > 0.5:
                results['successful_tracks'] += 1
            
            # Health score (0-1)
            health_score = 1.0 if health['overall_health'] == 'healthy' else 0.5
            results['system_health_scores'].append(health_score)
            
            # Compute pose error if ground truth available
            timestamp = float(rgb_file.stem)
            if timestamp in gt_poses:
                gt_pos = np.array(gt_poses[timestamp])
                est_pos = pose.position
                error = np.linalg.norm(gt_pos - est_pos)
                results['pose_errors'].append(error)
            
            if i % 10 == 0:
                print(f"  Frame {i:3d}: conf={pose.confidence:.2f}, "
                      f"dets={len(detections)}, time={performance['total_time_ms']:.1f}ms")
        
        except Exception as e:
            print(f"  Frame {i:3d}: FAILED - {e}")
            continue
    
    return results


def evaluate_detection_on_flir(
    flir_path: str,
    pipeline: MultiModalPipeline,
    max_images: int = 100
) -> Dict[str, Any]:
    """
    Evaluate detection performance on FLIR thermal dataset
    
    Args:
        flir_path: Path to FLIR dataset
        pipeline: Initialized pipeline
        max_images: Maximum images to process
        
    Returns:
        Detection metrics
    """
    flir_dir = Path(flir_path)
    thermal_test_dir = flir_dir / "video_thermal_test" / "data"
    
    if not thermal_test_dir.exists():
        raise ValueError(f"FLIR thermal test directory not found: {thermal_test_dir}")
    
    # Get thermal images
    thermal_files = sorted(list(thermal_test_dir.glob("*.jpg")))[:max_images]
    
    results = {
        'total_images': 0,
        'total_detections': 0,
        'detection_confidences': [],
        'processing_times': [],
        'class_counts': {}
    }
    
    print(f"Evaluating detection on {len(thermal_files)} FLIR thermal images")
    
    for i, thermal_file in enumerate(thermal_files):
        # Load thermal image
        thermal_frame = cv2.imread(str(thermal_file), cv2.IMREAD_GRAYSCALE)
        if thermal_frame is None:
            continue
        
        try:
            # Run detection only
            start_time = time.time()
            detections = pipeline.detection_module.detect_targets(thermal_frame)
            detection_time = (time.time() - start_time) * 1000
            
            results['total_images'] += 1
            results['total_detections'] += len(detections)
            results['processing_times'].append(detection_time)
            
            for det in detections:
                results['detection_confidences'].append(det.confidence)
                
                if det.class_name not in results['class_counts']:
                    results['class_counts'][det.class_name] = 0
                results['class_counts'][det.class_name] += 1
            
            if i % 20 == 0:
                print(f"  Image {i:3d}: {len(detections)} detections, "
                      f"{detection_time:.1f}ms")
        
        except Exception as e:
            print(f"  Image {i:3d}: FAILED - {e}")
            continue
    
    return results


def compute_summary_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary metrics from evaluation results"""
    summary = {}
    
    if results['total_frames'] > 0:
        summary['tracking_success_rate'] = results['successful_tracks'] / results['total_frames']
        summary['avg_slam_confidence'] = np.mean(results['slam_confidences'])
        summary['avg_processing_time_ms'] = np.mean(results['processing_times'])
        summary['avg_fps'] = 1000.0 / summary['avg_processing_time_ms']
        summary['avg_detections_per_frame'] = np.mean(results['detection_counts'])
        summary['avg_system_health_score'] = np.mean(results['system_health_scores'])
        
        if results['pose_errors']:
            summary['avg_pose_error_m'] = np.mean(results['pose_errors'])
            summary['pose_error_std_m'] = np.std(results['pose_errors'])
    
    return summary


def print_evaluation_report(results: Dict[str, Any], summary: Dict[str, float]):
    """Print formatted evaluation report"""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"Total Frames Processed: {results['total_frames']}")
    print(f"Successful Tracks: {results['successful_tracks']}")
    print(f"Tracking Success Rate: {summary.get('tracking_success_rate', 0):.2%}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average Processing Time: {summary.get('avg_processing_time_ms', 0):.1f} ms")
    print(f"  Average FPS: {summary.get('avg_fps', 0):.1f}")
    print(f"  Average SLAM Confidence: {summary.get('avg_slam_confidence', 0):.2f}")
    print(f"  Average System Health Score: {summary.get('avg_system_health_score', 0):.2f}")
    
    if 'avg_pose_error_m' in summary:
        print(f"\nSLAM Accuracy:")
        print(f"  Average Pose Error: {summary['avg_pose_error_m']:.3f} m")
        print(f"  Pose Error Std Dev: {summary['pose_error_std_m']:.3f} m")
    
    print(f"\nDetection Metrics:")
    print(f"  Average Detections per Frame: {summary.get('avg_detections_per_frame', 0):.1f}")
    print(f"  Total Detections: {results.get('total_detections', 0)}")
    
    if 'detection_confidences' in results and results['detection_confidences']:
        avg_det_conf = np.mean(results['detection_confidences'])
        print(f"  Average Detection Confidence: {avg_det_conf:.2f}")
    
    if 'class_counts' in results:
        print(f"\nDetection Class Distribution:")
        for class_name, count in results['class_counts'].items():
            print(f"  {class_name}: {count}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate thermal SLAM system")
    parser.add_argument(
        '--tum-dataset',
        type=str,
        help='Path to TUM RGB-D dataset for SLAM evaluation'
    )
    parser.add_argument(
        '--flir-dataset', 
        type=str,
        help='Path to FLIR dataset for detection evaluation'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum frames to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for detailed results'
    )
    
    args = parser.parse_args()
    
    if not args.tum_dataset and not args.flir_dataset:
        print("Error: Must specify either --tum-dataset or --flir-dataset")
        return
    
    # Initialize pipeline
    print("Initializing thermal SLAM pipeline...")
    pipeline = MultiModalPipeline()
    
    all_results = {}
    
    # Evaluate on TUM dataset
    if args.tum_dataset:
        print(f"\nEvaluating SLAM on TUM dataset: {args.tum_dataset}")
        tum_results = evaluate_on_tum_dataset(
            args.tum_dataset, pipeline, args.max_frames
        )
        all_results['tum_slam'] = tum_results
        
        tum_summary = compute_summary_metrics(tum_results)
        print_evaluation_report(tum_results, tum_summary)
    
    # Evaluate on FLIR dataset
    if args.flir_dataset:
        print(f"\nEvaluating detection on FLIR dataset: {args.flir_dataset}")
        flir_results = evaluate_detection_on_flir(
            args.flir_dataset, pipeline, args.max_frames
        )
        all_results['flir_detection'] = flir_results
        
        flir_summary = compute_summary_metrics(flir_results)
        print_evaluation_report(flir_results, flir_summary)
    
    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Clean results for JSON
        json_results = {}
        for key, value in all_results.items():
            json_results[key] = {k: convert_numpy(v) for k, v in value.items()}
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")
    
    # Shutdown pipeline
    pipeline.shutdown()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
