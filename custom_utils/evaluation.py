import numpy as np
from difflib import SequenceMatcher

def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in normalized format: (x_center, y_center, width, height).
    """
    def to_corners(box):
        x, y, w, h = box
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return x1, y1, x2, y2

    x1A, y1A, x2A, y2A = to_corners(boxA)
    x1B, y1B, x2B, y2B = to_corners(boxB)
    
    x_left = max(x1A, x1B)
    y_top = max(y1A, y1B)
    x_right = min(x2A, x2B)
    y_bottom = min(y2A, y2B)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = (x2A - x1A) * (y2A - y1A)
    boxB_area = (x2B - x1B) * (y2B - y1B)
    
    return inter_area / float(boxA_area + boxB_area - inter_area)

def compute_map(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP) for detection.
    Each box is represented as (class, x, y, w, h).
    """
    true_positives = 0
    for pred in pred_boxes:
        for gt in gt_boxes:
            if pred[0] == gt[0]:
                iou = compute_iou(pred[1:], gt[1:])
                if iou >= iou_threshold:
                    true_positives += 1
                    break
    precision = true_positives / (len(pred_boxes) + 1e-6)
    recall = true_positives / (len(gt_boxes) + 1e-6)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_CER(pred_str, gt_str):
    """
    Compute the Character Error Rate (CER) using Levenshtein distance.
    """
    matcher = SequenceMatcher(None, pred_str, gt_str)
    opcodes = matcher.get_opcodes()
    errors = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'equal':
            errors += max(i2 - i1, j2 - j1)
    return errors / max(len(gt_str), 1)
