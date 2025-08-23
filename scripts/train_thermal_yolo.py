"""
Training script for thermal YOLO detection
Fine-tunes YOLOv8 on FLIR ADAS thermal dataset
"""
import os
import yaml
import json
from pathlib import Path
from ultralytics import YOLO
import argparse


def create_dataset_config(flir_path: str, output_path: str = "data/thermal_dataset.yaml"):
    """
    Create YOLO dataset configuration for FLIR ADAS thermal data
    
    Args:
        flir_path: Path to FLIR_ADAS_v2 dataset
        output_path: Output path for dataset config
    """
    flir_dir = Path(flir_path)
    
    # FLIR ADAS class mapping
    class_names = [
        'person', 'bike', 'car', 'motor', 'bus', 'train', 'truck', 'light'
    ]
    
    # Dataset configuration
    config = {
        'path': str(flir_dir.absolute()),
        'train': 'images_thermal_train/data',
        'val': 'images_thermal_val/data',
        'test': 'video_thermal_test/data',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset configuration created: {output_path}")
    return output_path


def convert_coco_to_yolo(coco_json_path: str, images_dir: str, output_dir: str):
    """
    Convert COCO annotations to YOLO format
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO labels
    """
    import json
    from pathlib import Path
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create image ID to filename mapping
    image_info = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Convert each image's annotations
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_info:
            continue
            
        filename = image_info[image_id]
        image_path = Path(images_dir) / filename
        
        if not image_path.exists():
            continue
        
        # Get image dimensions (assuming from COCO data)
        image_width = None
        image_height = None
        for img in coco_data['images']:
            if img['id'] == image_id:
                image_width = img['width']
                image_height = img['height']
                break
        
        if not image_width or not image_height:
            continue
        
        # Convert annotations to YOLO format
        yolo_annotations = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            center_x = (bbox[0] + bbox[2] / 2) / image_width
            center_y = (bbox[1] + bbox[3] / 2) / image_height
            norm_width = bbox[2] / image_width
            norm_height = bbox[3] / image_height
            
            yolo_annotations.append(f"{category_id} {center_x} {center_y} {norm_width} {norm_height}")
        
        # Write YOLO label file
        label_filename = Path(filename).stem + '.txt'
        label_path = output_path / label_filename
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    print(f"Converted {len(annotations_by_image)} images to YOLO format in {output_dir}")


def prepare_flir_dataset(flir_path: str):
    """
    Prepare FLIR dataset for YOLO training
    
    Args:
        flir_path: Path to FLIR_ADAS_v2 dataset
    """
    flir_dir = Path(flir_path)
    
    # Convert training annotations
    train_coco = flir_dir / "images_thermal_train" / "coco.json"
    train_images = flir_dir / "images_thermal_train" / "data"
    train_labels = flir_dir / "images_thermal_train" / "labels"
    
    if train_coco.exists():
        convert_coco_to_yolo(str(train_coco), str(train_images), str(train_labels))
    
    # Convert validation annotations
    val_coco = flir_dir / "images_thermal_val" / "coco.json"
    val_images = flir_dir / "images_thermal_val" / "data"
    val_labels = flir_dir / "images_thermal_val" / "labels"
    
    if val_coco.exists():
        convert_coco_to_yolo(str(val_coco), str(val_images), str(val_labels))
    
    print("FLIR dataset preparation complete")


def train_thermal_yolo(
    dataset_config: str,
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "auto"
):
    """
    Train thermal YOLO model
    
    Args:
        dataset_config: Path to dataset YAML config
        model_size: YOLO model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        device: Training device
    """
    # Load pretrained YOLO model
    model_name = f"yolov8{model_size}.pt"
    model = YOLO(model_name)
    
    print(f"Training {model_name} on thermal data...")
    print(f"Dataset: {dataset_config}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    
    # Train the model
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="thermal_yolo_training",
        name=f"thermal_yolo_{model_size}",
        save=True,
        plots=True,
        verbose=True
    )
    
    # Validate the model
    validation_results = model.val()
    
    print("Training completed!")
    print(f"Best model saved to: {results.save_dir}")
    print(f"Validation mAP@0.5: {validation_results.box.map50:.3f}")
    print(f"Validation mAP@0.5:0.95: {validation_results.box.map:.3f}")
    
    return results


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train thermal YOLO model")
    parser.add_argument(
        '--flir-path',
        type=str,
        required=True,
        help='Path to FLIR_ADAS_v2 dataset directory'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO model size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Training device (auto, cpu, cuda, mps)'
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare dataset, do not train'
    )
    
    args = parser.parse_args()
    
    # Check if FLIR dataset exists
    flir_path = Path(args.flir_path)
    if not flir_path.exists():
        print(f"Error: FLIR dataset path does not exist: {flir_path}")
        return
    
    print(f"Using FLIR dataset: {flir_path}")
    
    # Prepare dataset
    print("Preparing FLIR dataset...")
    prepare_flir_dataset(str(flir_path))
    
    # Create dataset configuration
    dataset_config = create_dataset_config(str(flir_path))
    
    if args.prepare_only:
        print("Dataset preparation complete. Use --train to start training.")
        return
    
    # Train model
    print("Starting training...")
    results = train_thermal_yolo(
        dataset_config=dataset_config,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device
    )
    
    print(f"Training complete! Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
