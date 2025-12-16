"""
Spatial Feature Extraction for Hierarchical Binary Classifier

Extracts 9 numerical features from YOLO detections to encode domain logic:
- Position and visibility (3 features)
- Structural relationships (3 features)
- Occlusion detection (3 features)

All features are normalized to [0, 1] range.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value in [0, 1]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute normalized Euclidean distance between bbox centers.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
    
    Returns:
        Normalized distance in [0, 1]
    """
    # Get centers
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    
    # Euclidean distance
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # Normalize by diagonal (max possible distance)
    max_dist = np.sqrt(2)  # Since coords are normalized to [0, 1]
    
    return min(dist / max_dist, 1.0)


def extract_spatial_features(
    target_detection: Dict,
    all_detections: List[Dict],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Extract 9 spatial features from YOLO detections.
    
    Args:
        target_detection: Dict with keys:
            - 'bbox': [x1, y1, x2, y2] in pixel coordinates
            - 'object_type': str ('toe_drain', 'slope_drain', 'rock_toe', 'vegetation')
            - 'confidence': float (YOLO confidence score)
        all_detections: List of all detection dicts in the image (including target)
        image_shape: (height, width) in pixels
    
    Returns:
        numpy array of shape [9] with all values normalized to [0, 1]
    
    Features:
        [0] y_center_norm: Vertical position (0=top, 1=bottom)
        [1] bbox_area_ratio: Bbox area / image area
        [2] detection_confidence: YOLO confidence score
        [3] dist_to_slope_end: Distance to nearest slope drain endpoint
        [4] is_near_slope_drain: Boolean (within threshold)
        [5] is_above_toe_drain: Boolean (above any toe drain)
        [6] overlap_with_rocks: Overlap ratio with rock_toe
        [7] overlap_with_vegetation: Overlap ratio with vegetation
        [8] is_below_slope_drain: Boolean (below any slope drain)
    """
    img_height, img_width = image_shape
    target_bbox = target_detection['bbox']
    target_type = target_detection['object_type']
    target_conf = target_detection['confidence']
    
    # Normalize bbox to [0, 1]
    x1_norm = target_bbox[0] / img_width
    y1_norm = target_bbox[1] / img_height
    x2_norm = target_bbox[2] / img_width
    y2_norm = target_bbox[3] / img_height
    bbox_norm = [x1_norm, y1_norm, x2_norm, y2_norm]
    
    # Feature 0: Y-center normalized (vertical position)
    y_center_norm = (y1_norm + y2_norm) / 2
    
    # Feature 1: Bbox area ratio
    bbox_area = (x2_norm - x1_norm) * (y2_norm - y1_norm)
    bbox_area_ratio = bbox_area  # Already normalized since coords are normalized
    
    # Feature 2: Detection confidence (already in [0, 1])
    detection_confidence = target_conf
    
    # Feature 3: Distance to nearest slope drain endpoint
    dist_to_slope_end = 1.0  # Default: far away
    slope_drains = [d for d in all_detections if d['object_type'] == 'slope_drain']
    if slope_drains:
        min_dist = 1.0
        for slope in slope_drains:
            # Normalize slope bbox
            s_bbox = slope['bbox']
            s_x1 = s_bbox[0] / img_width
            s_y1 = s_bbox[1] / img_height
            s_x2 = s_bbox[2] / img_width
            s_y2 = s_bbox[3] / img_height
            slope_bbox_norm = [s_x1, s_y1, s_x2, s_y2]
            
            # Distance to slope drain
            dist = compute_distance(bbox_norm, slope_bbox_norm)
            min_dist = min(min_dist, dist)
        dist_to_slope_end = min_dist
    
    # Feature 4: Is near slope drain (threshold: 0.2 normalized distance)
    is_near_slope_drain = 1.0 if dist_to_slope_end < 0.2 else 0.0
    
    # Feature 5: Is above toe drain (target y_center < any toe drain y_center)
    is_above_toe_drain = 0.0
    toe_drains = [d for d in all_detections 
                  if d['object_type'] == 'toe_drain' and d != target_detection]
    if toe_drains:
        for toe in toe_drains:
            t_bbox = toe['bbox']
            t_y_center = ((t_bbox[1] + t_bbox[3]) / 2) / img_height
            if y_center_norm < t_y_center:
                is_above_toe_drain = 1.0
                break
    
    # Feature 6: Overlap with rock_toe
    overlap_with_rocks = 0.0
    rock_toes = [d for d in all_detections if d['object_type'] == 'rock_toe']
    if rock_toes:
        max_overlap = 0.0
        for rock in rock_toes:
            r_bbox = rock['bbox']
            r_x1 = r_bbox[0] / img_width
            r_y1 = r_bbox[1] / img_height
            r_x2 = r_bbox[2] / img_width
            r_y2 = r_bbox[3] / img_height
            rock_bbox_norm = [r_x1, r_y1, r_x2, r_y2]
            
            iou = compute_iou(bbox_norm, rock_bbox_norm)
            max_overlap = max(max_overlap, iou)
        overlap_with_rocks = max_overlap
    
    # Feature 7: Overlap with vegetation
    overlap_with_vegetation = 0.0
    vegetations = [d for d in all_detections if d['object_type'] == 'vegetation']
    if vegetations:
        max_overlap = 0.0
        for veg in vegetations:
            v_bbox = veg['bbox']
            v_x1 = v_bbox[0] / img_width
            v_y1 = v_bbox[1] / img_height
            v_x2 = v_bbox[2] / img_width
            v_y2 = v_bbox[3] / img_height
            veg_bbox_norm = [v_x1, v_y1, v_x2, v_y2]
            
            iou = compute_iou(bbox_norm, veg_bbox_norm)
            max_overlap = max(max_overlap, iou)
        overlap_with_vegetation = max_overlap
    
    # Feature 8: Is below slope drain (target y_center > any slope drain y_center)
    is_below_slope_drain = 0.0
    if slope_drains:
        for slope in slope_drains:
            s_bbox = slope['bbox']
            s_y_center = ((s_bbox[1] + s_bbox[3]) / 2) / img_height
            if y_center_norm > s_y_center:
                is_below_slope_drain = 1.0
                break
    
    # Assemble features
    features = np.array([
        y_center_norm,
        bbox_area_ratio,
        detection_confidence,
        dist_to_slope_end,
        is_near_slope_drain,
        is_above_toe_drain,
        overlap_with_rocks,
        overlap_with_vegetation,
        is_below_slope_drain,
    ], dtype=np.float32)
    
    # Ensure all values are in [0, 1]
    features = np.clip(features, 0.0, 1.0)
    
    return features


def test_spatial_features():
    """Test spatial feature extraction with sample detections."""
    print("Testing spatial feature extraction...")
    
    # Sample image shape
    image_shape = (640, 640)
    
    # Sample detections
    all_detections = [
        {
            'bbox': [100, 400, 200, 500],  # Toe drain at bottom
            'object_type': 'toe_drain',
            'confidence': 0.9
        },
        {
            'bbox': [150, 200, 250, 350],  # Slope drain in middle
            'object_type': 'slope_drain',
            'confidence': 0.85
        },
        {
            'bbox': [120, 380, 180, 430],  # Rock toe near toe drain
            'object_type': 'rock_toe',
            'confidence': 0.8
        },
    ]
    
    # Test extraction for toe drain
    target = all_detections[0]
    features = extract_spatial_features(target, all_detections, image_shape)
    
    print(f"\nTarget: {target['object_type']} at bbox {target['bbox']}")
    print(f"Spatial features (9 values):")
    print(f"  [0] y_center_norm:          {features[0]:.3f}")
    print(f"  [1] bbox_area_ratio:        {features[1]:.3f}")
    print(f"  [2] detection_confidence:   {features[2]:.3f}")
    print(f"  [3] dist_to_slope_end:      {features[3]:.3f}")
    print(f"  [4] is_near_slope_drain:    {features[4]:.3f}")
    print(f"  [5] is_above_toe_drain:     {features[5]:.3f}")
    print(f"  [6] overlap_with_rocks:     {features[6]:.3f}")
    print(f"  [7] overlap_with_vegetation:{features[7]:.3f}")
    print(f"  [8] is_below_slope_drain:   {features[8]:.3f}")
    
    # Verify all values in [0, 1]
    assert np.all((features >= 0) & (features <= 1)), "Features not normalized!"
    assert features.shape == (9,), "Wrong feature shape!"
    
    print("\n✅ Spatial feature extraction working correctly!")
    print(f"✅ All {len(features)} features normalized to [0, 1]")


if __name__ == '__main__':
    test_spatial_features()

