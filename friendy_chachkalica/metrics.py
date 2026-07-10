from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch


DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def evaluate_detection(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[Dict[str, Any]],
    iou_thresholds: Optional[Iterable[float]] = None,
    score_threshold: float = 0.001,
    map_score_threshold: Optional[float] = None,
    num_classes: Optional[int] = None,
    prediction_classes: Optional[Dict[int, str]] = None,
    target_classes: Optional[Dict[int, str]] = None,
    eval_classes: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Evaluate Friendy-format detection predictions against target dicts.

    When eval_classes is provided, predictions and targets are remapped by class
    name into that evaluation class space. Classes absent from eval_classes are
    ignored, which lets a model trained on extra classes be compared on only the
    classes present in the validation or test dataset.
    """
    thresholds = [float(value) for value in (iou_thresholds or DEFAULT_IOU_THRESHOLDS)]
    if not thresholds:
        raise ValueError('iou_thresholds must contain at least one threshold')

    operating_threshold = float(score_threshold)
    ap_score_threshold = operating_threshold if map_score_threshold is None else float(map_score_threshold)

    prepared_targets = [_prepare_target(target) for target in targets]
    prepared_predictions = [
        _prepare_prediction(prediction, target, ap_score_threshold)
        for prediction, target in zip(predictions, prepared_targets)
    ]

    # When both eval_classes and prediction_classes are given, restrict evaluation
    # to the intersection by name so classes the model was never trained on don't
    # count as AP=0 in the average.
    effective_eval_classes = eval_classes
    if eval_classes is not None and prediction_classes is not None:
        prediction_names = {str(name) for name in prediction_classes.values()}
        effective_eval_classes = {
            class_id: name
            for class_id, name in eval_classes.items()
            if str(name) in prediction_names
        }

    class_ids = _resolve_class_ids(
        prepared_predictions,
        prepared_targets,
        num_classes,
        eval_classes=effective_eval_classes,
    )
    if effective_eval_classes is not None:
        prepared_predictions, prepared_targets = _remap_to_eval_classes(
            prepared_predictions,
            prepared_targets,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=effective_eval_classes,
        )

    ap_by_threshold = {}
    precision_by_class = {}
    recall_by_class = {}
    f1_by_class = {}
    gt_count_by_class = {}
    pred_count_by_class = {}

    for threshold in thresholds:
        ap_by_threshold[threshold] = {}

    operating_prediction_count = 0
    operating_pred_count_by_class = {class_id: 0 for class_id in class_ids}
    for prediction in prepared_predictions:
        keep = prediction['scores'] >= operating_threshold
        operating_prediction_count += int(keep.sum())
        for class_id in class_ids:
            operating_pred_count_by_class[class_id] += int(
                (keep & (prediction['labels'] == class_id)).sum()
            )

    # AP/mAP integrates the ranked detections down to ap_score_threshold.
    # Precision/recall/F1 and the confusion matrix are single operating-point
    # metrics, so they use score_threshold instead.
    micro = _micro_stats(
        prepared_predictions,
        prepared_targets,
        iou_threshold=0.5,
        score_cut=operating_threshold,
    )

    # box_iou between each prediction and its image's ground truth does not depend
    # on the IoU threshold, so precompute the best match per prediction once per
    # class and reuse it across every threshold.
    for class_id in class_ids:
        precomputed = _precompute_class_matching(
            prepared_predictions,
            prepared_targets,
            class_id=class_id,
        )
        gt_count_by_class[class_id] = precomputed['gt_count']
        pred_count_by_class[class_id] = operating_pred_count_by_class.get(class_id, 0)
        for threshold in thresholds:
            stats = _class_stats_at_iou(
                precomputed,
                iou_threshold=threshold,
                score_cut=operating_threshold,
            )
            ap_by_threshold[threshold][class_id] = stats['ap']
            if threshold == 0.5:
                precision_by_class[class_id] = stats['precision']
                recall_by_class[class_id] = stats['recall']
                f1_by_class[class_id] = _f1(stats['precision'], stats['recall'])

    ap50_by_class = ap_by_threshold.get(0.5, {})
    ap5095_by_class = {
        class_id: _mean([ap_by_threshold[threshold][class_id] for threshold in thresholds])
        for class_id in class_ids
    }

    # A class declared in the eval space but absent from the ground truth has a
    # hard AP of 0, so averaging over it silently deflates mAP (a perfect
    # detector on a dataset whose classes.txt lists unused classes scores < 1).
    # The mAP means therefore only average classes with ground-truth instances;
    # the rest stay visible in per_class with ground_truth_count=0.
    classes_with_gt = [
        class_id for class_id in class_ids if gt_count_by_class.get(class_id, 0) > 0
    ]

    confusion_iou = 0.5 if 0.5 in thresholds else thresholds[0]
    confusion = _confusion_matrix(
        prepared_predictions,
        prepared_targets,
        class_ids,
        effective_eval_classes,
        iou_threshold=confusion_iou,
        conf_threshold=operating_threshold,
    )

    return {
        'map50': _mean([ap50_by_class[class_id] for class_id in classes_with_gt if class_id in ap50_by_class]),
        'map50_95': _mean([ap5095_by_class[class_id] for class_id in classes_with_gt]),
        'precision': micro['precision'],
        'recall': micro['recall'],
        'f1': micro['f1'],
        'f1_confidence': micro['f1_confidence'],
        'num_eval_classes': len(class_ids),
        'num_eval_classes_with_gt': len(classes_with_gt),
        'num_images': len(targets),
        'num_predictions': operating_prediction_count,
        'num_targets': int(sum(len(target['labels']) for target in prepared_targets)),
        'iou_thresholds': thresholds,
        'confusion_matrix': confusion,
        'per_class': {
            int(class_id): {
                'class_name': _class_name(effective_eval_classes, class_id),
                'ap50': ap50_by_class.get(class_id, 0.0),
                'ap50_95': ap5095_by_class.get(class_id, 0.0),
                'precision': precision_by_class.get(class_id, 0.0),
                'recall': recall_by_class.get(class_id, 0.0),
                'f1': f1_by_class.get(class_id, 0.0),
                'ground_truth_count': gt_count_by_class.get(class_id, 0),
                'prediction_count': pred_count_by_class.get(class_id, 0),
            }
            for class_id in class_ids
        },
    }


HARD_IMAGE_METRIC = "detection_error_count"
HARD_IMAGE_METRIC_DESCRIPTION = (
    "difficulty = missed_GT + false_positives + wrong_class_matches "
    "+ Σ(1 - IoU over correctly-classified matches); higher = worse"
)


def select_hard_images(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[Dict[str, Any]],
    image_infos: Sequence[Dict[str, Any]],
    *,
    top_k: int = 50,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
    prediction_classes: Optional[Dict[int, str]] = None,
    target_classes: Optional[Dict[int, str]] = None,
    eval_classes: Optional[Dict[int, str]] = None,
    max_display_predictions: Optional[int] = 20,
) -> list[Dict[str, Any]]:
    """Rank images by a per-image detection-error score and return the worst ``top_k``.

    The score is :data:`HARD_IMAGE_METRIC_DESCRIPTION`. Predictions and ground truth are
    prepared and remapped into the eval class space exactly like :func:`evaluate_detection`,
    so the boxes returned (normalized center-xywh, with resolved class names) match the score.
    Each entry is self-contained for a viewer: image path/name, difficulty + component
    breakdown, and prediction/ground-truth boxes. Displayed predictions are capped by
    confidence so low-threshold artifacts stay readable in the browser.
    """
    prepared_targets = [_prepare_target(target) for target in targets]
    prepared_predictions = [
        _prepare_prediction(prediction, target, float(score_threshold))
        for prediction, target in zip(predictions, prepared_targets)
    ]

    # Mirror evaluate_detection's eval-class remap so class ids/names match the metrics.
    effective_eval_classes = eval_classes
    if eval_classes is not None and prediction_classes is not None:
        prediction_names = {str(name) for name in prediction_classes.values()}
        effective_eval_classes = {
            class_id: name
            for class_id, name in eval_classes.items()
            if str(name) in prediction_names
        }
    if effective_eval_classes is not None:
        prepared_predictions, prepared_targets = _remap_to_eval_classes(
            prepared_predictions,
            prepared_targets,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=effective_eval_classes,
        )

    name_lookup = (
        _normalize_class_map(effective_eval_classes)
        if effective_eval_classes is not None
        else None
    )

    scored = []
    for index, (prediction, target) in enumerate(zip(prepared_predictions, prepared_targets)):
        match = match_image(prediction, target, float(iou_threshold))
        wrong_class = sum(1 for pair in match['matches'] if not pair['class_correct'])
        loc_error = sum(1.0 - pair['iou'] for pair in match['matches'] if pair['class_correct'])
        missed = len(match['misses'])
        false_positives = len(match['false_positives'])
        difficulty = missed + false_positives + wrong_class + loc_error

        info = image_infos[index] if index < len(image_infos) and isinstance(image_infos[index], dict) else {}
        image_path = info.get('image_path')
        height, width = [int(value) for value in target['orig_size']]
        scored.append({
            'image_path': str(image_path) if image_path else None,
            'image_name': Path(str(image_path)).name if image_path else f'image_{index}',
            'difficulty': round(float(difficulty), 4),
            'missed': missed,
            'false_positives': false_positives,
            'wrong_class': wrong_class,
            'loc_error': round(float(loc_error), 4),
            'num_predictions': int(prediction['labels'].numel()),
            'num_ground_truth': int(target['labels'].numel()),
            'predictions': _boxes_for_display(
                _limit_predictions_for_display(prediction, max_display_predictions),
                width,
                height,
                name_lookup,
                with_score=True,
            ),
            'ground_truth': _boxes_for_display(target, width, height, name_lookup, with_score=False),
        })

    scored.sort(key=lambda entry: entry['difficulty'], reverse=True)
    return scored[:int(top_k)]


def _limit_predictions_for_display(detection, max_predictions: Optional[int]):
    if max_predictions is None or int(max_predictions) <= 0:
        return detection
    if detection['scores'].numel() <= int(max_predictions):
        return detection
    keep = torch.argsort(detection['scores'], descending=True, stable=True)[: int(max_predictions)]
    return {
        'boxes': detection['boxes'][keep],
        'scores': detection['scores'][keep],
        'labels': detection['labels'][keep],
    }


def _boxes_for_display(detection, image_width, image_height, name_lookup, with_score):
    """Convert an xyxy-pixel detection dict to normalized center-xywh box dicts for a viewer."""
    boxes = detection['boxes']
    labels = detection['labels']
    scores = detection.get('scores')
    if boxes.numel() == 0:
        return []
    width = max(int(image_width), 1)
    height = max(int(image_height), 1)
    result = []
    for row in range(boxes.shape[0]):
        x1, y1, x2, y2 = (float(value) for value in boxes[row])
        class_id = int(labels[row])
        name = (name_lookup.get(class_id) if name_lookup else None) or str(class_id)
        box = {
            'cx': ((x1 + x2) / 2) / width,
            'cy': ((y1 + y2) / 2) / height,
            'w': (x2 - x1) / width,
            'h': (y2 - y1) / height,
            'class_id': class_id,
            'class_name': name,
        }
        if with_score and scores is not None:
            box['confidence'] = round(float(scores[row]), 4)
        result.append(box)
    return result


def _prepare_target(target: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    boxes = target.get('boxes', torch.empty((0, 4)))
    labels = target.get('labels', torch.empty((0,), dtype=torch.long))
    orig_size = target.get('orig_size')
    if orig_size is None:
        if boxes.numel() == 0:
            height, width = 1, 1
        else:
            width = int(torch.max(boxes[:, 2]).item())
            height = int(torch.max(boxes[:, 3]).item())
    elif torch.is_tensor(orig_size):
        height, width = [int(value) for value in orig_size.detach().cpu().flatten()[:2]]
    else:
        height, width = [int(value) for value in orig_size[:2]]

    return {
        'boxes': boxes.detach().cpu().float().reshape(-1, 4),
        'labels': labels.detach().cpu().long().reshape(-1),
        'orig_size': torch.tensor([height, width], dtype=torch.long),
    }


def _prepare_prediction(
    prediction: torch.Tensor,
    target: Dict[str, torch.Tensor],
    score_threshold: float,
) -> Dict[str, torch.Tensor]:
    if prediction is None or prediction.numel() == 0:
        return _empty_prediction()

    prediction = prediction.detach().cpu().float().reshape(-1, 6)
    prediction = prediction[prediction[:, 4] >= float(score_threshold)]
    if prediction.numel() == 0:
        return _empty_prediction()

    height, width = [int(value) for value in target['orig_size']]
    boxes = _xywhn_to_xyxy_tensor(prediction[:, :4], image_width=width, image_height=height)
    return {
        'boxes': boxes,
        'scores': prediction[:, 4],
        'labels': prediction[:, 5].long(),
    }


def _empty_prediction() -> Dict[str, torch.Tensor]:
    return {
        'boxes': torch.empty((0, 4), dtype=torch.float32),
        'scores': torch.empty((0,), dtype=torch.float32),
        'labels': torch.empty((0,), dtype=torch.long),
    }


def _resolve_class_ids(
    predictions,
    targets,
    num_classes: Optional[int],
    eval_classes: Optional[Dict[int, str]] = None,
) -> list[int]:
    if eval_classes is not None:
        return sorted(int(class_id) for class_id in eval_classes)
    if num_classes is not None:
        return list(range(int(num_classes)))

    class_ids = set()
    for target in targets:
        class_ids.update(int(value) for value in target['labels'].tolist())
    for prediction in predictions:
        class_ids.update(int(value) for value in prediction['labels'].tolist())
    return sorted(class_ids)


def _remap_to_eval_classes(
    predictions,
    targets,
    prediction_classes: Optional[Dict[int, str]],
    target_classes: Optional[Dict[int, str]],
    eval_classes: Dict[int, str],
):
    eval_name_to_id = {str(name): int(class_id) for class_id, name in eval_classes.items()}
    prediction_id_to_name = _normalize_class_map(prediction_classes)
    target_id_to_name = _normalize_class_map(target_classes)

    remapped_predictions = [
        _remap_prediction(prediction, prediction_id_to_name, eval_name_to_id)
        for prediction in predictions
    ]
    remapped_targets = [
        _remap_target(target, target_id_to_name, eval_name_to_id)
        for target in targets
    ]
    return remapped_predictions, remapped_targets


def _normalize_class_map(class_map: Optional[Dict[int, str]]) -> Optional[Dict[int, str]]:
    if class_map is None:
        return None
    return {int(class_id): str(name) for class_id, name in class_map.items()}


def _remap_prediction(prediction, id_to_name, eval_name_to_id):
    if id_to_name is None:
        return prediction

    kept_boxes = []
    kept_scores = []
    kept_labels = []
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        class_name = id_to_name.get(int(label))
        if class_name not in eval_name_to_id:
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
        kept_labels.append(eval_name_to_id[class_name])

    return _build_remapped_detection(prediction, kept_boxes, kept_scores, kept_labels)


def _remap_target(target, id_to_name, eval_name_to_id):
    if id_to_name is None:
        return target

    kept_boxes = []
    kept_labels = []
    for box, label in zip(target['boxes'], target['labels']):
        class_name = id_to_name.get(int(label))
        if class_name not in eval_name_to_id:
            continue
        kept_boxes.append(box)
        kept_labels.append(eval_name_to_id[class_name])

    boxes = torch.stack(kept_boxes) if kept_boxes else target['boxes'].new_zeros((0, 4))
    labels = torch.tensor(kept_labels, dtype=torch.long)
    return {
        'boxes': boxes,
        'labels': labels,
        'orig_size': target['orig_size'],
    }


def _build_remapped_detection(reference, boxes, scores, labels):
    if not boxes:
        return _empty_prediction()
    return {
        'boxes': torch.stack(boxes),
        'scores': torch.stack(scores).float(),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def _class_name(eval_classes: Optional[Dict[int, str]], class_id: int) -> Optional[str]:
    if eval_classes is None:
        return None
    class_name = eval_classes.get(int(class_id))
    if class_name is None:
        return None
    return str(class_name)


def _precompute_class_matching(predictions, targets, class_id: int) -> Dict[str, Any]:
    """Precompute the threshold-independent best ground-truth match per prediction.

    Predictions of ``class_id`` are gathered across all images and sorted by score
    descending (stable, to reproduce the original list.sort(reverse=True) tie order).
    For each prediction we compute, with a single batched box_iou per image, the
    best-overlapping ground-truth box in its own image and that overlap value. The
    IoU-threshold comparison and the greedy one-gt-per-prediction matching are then
    applied per threshold in ``_class_stats_at_iou``.
    """
    gt_by_image = []
    scores_parts = []
    boxes_parts = []
    image_parts = []
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        gt_by_image.append(target['boxes'][target['labels'] == class_id])

        mask = prediction['labels'] == class_id
        count = int(mask.sum())
        if count == 0:
            continue
        scores_parts.append(prediction['scores'][mask])
        boxes_parts.append(prediction['boxes'][mask])
        image_parts.append(torch.full((count,), image_index, dtype=torch.long))

    gt_count = int(sum(len(boxes) for boxes in gt_by_image))
    if not scores_parts:
        return {'pred_count': 0, 'gt_count': gt_count}

    scores = torch.cat(scores_parts)
    boxes = torch.cat(boxes_parts)
    image_index_per_pred = torch.cat(image_parts)

    order = torch.argsort(scores, descending=True, stable=True)
    scores = scores[order]
    boxes = boxes[order]
    image_index_per_pred = image_index_per_pred[order]
    pred_count = int(scores.numel())

    best_iou = torch.zeros((pred_count,), dtype=torch.float32)
    best_gt = torch.zeros((pred_count,), dtype=torch.long)
    has_gt = torch.zeros((pred_count,), dtype=torch.bool)

    for image_index in torch.unique(image_index_per_pred).tolist():
        gt_boxes = gt_by_image[image_index]
        if len(gt_boxes) == 0:
            continue
        select = (image_index_per_pred == image_index).nonzero(as_tuple=False).flatten()
        ious = box_iou(boxes[select], gt_boxes)
        per_best_iou, per_best_gt = torch.max(ious, dim=1)
        best_iou[select] = per_best_iou
        best_gt[select] = per_best_gt
        has_gt[select] = True

    # Encode (image, matched gt) into a single key so a ground-truth box can only
    # be claimed once. Stride guarantees keys never collide across images.
    max_gt_per_image = max((len(boxes_i) for boxes_i in gt_by_image), default=0)
    keys = image_index_per_pred * (max_gt_per_image + 1) + best_gt

    return {
        'pred_count': pred_count,
        'gt_count': gt_count,
        'scores': scores,
        'best_iou': best_iou,
        'has_gt': has_gt,
        'keys': keys,
    }


def _class_stats_at_iou(
    precomputed: Dict[str, Any],
    iou_threshold: float,
    score_cut: Optional[float] = None,
) -> Dict[str, Any]:
    gt_count = precomputed['gt_count']
    pred_count = precomputed['pred_count']
    if pred_count == 0:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'gt_count': gt_count, 'pred_count': 0}

    # A prediction is a true positive when its best gt overlaps at >= threshold and
    # it is the highest-scoring prediction claiming that gt; later collisions and
    # below-threshold predictions are false positives. Predictions are already in
    # descending-score order, so the lowest index per key wins the match.
    valid = precomputed['has_gt'] & (precomputed['best_iou'] >= iou_threshold)
    true_positives = torch.zeros((pred_count,), dtype=torch.float32)
    if bool(valid.any()):
        valid_index = valid.nonzero(as_tuple=False).flatten()
        first_local = _first_occurrence_mask(precomputed['keys'][valid_index])
        true_positives[valid_index[first_local]] = 1.0

    false_positives = 1.0 - true_positives
    tp_cumsum = torch.cumsum(true_positives, dim=0)
    fp_cumsum = torch.cumsum(false_positives, dim=0)
    precision_curve = tp_cumsum / torch.clamp(tp_cumsum + fp_cumsum, min=1e-12)
    recall_curve = tp_cumsum / max(gt_count, 1)

    # Precision/recall are reported at the operating confidence (score_cut);
    # AP always integrates the full curve. Matching is greedy in descending
    # score order, so truncating low-confidence predictions cannot change which
    # higher-scored prediction won each ground-truth box.
    if score_cut is None:
        cut = pred_count
    else:
        cut = int((precomputed['scores'] >= float(score_cut)).sum())

    if cut == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = float(precision_curve[cut - 1].item())
        recall = float(recall_curve[cut - 1].item()) if gt_count > 0 else 0.0

    return {
        'ap': _average_precision(recall_curve, precision_curve) if gt_count > 0 else 0.0,
        'precision': precision,
        'recall': recall,
        'gt_count': gt_count,
        'pred_count': pred_count,
    }


def _first_occurrence_mask(keys: torch.Tensor) -> torch.Tensor:
    """Boolean mask marking the first occurrence of each value, preserving input order.

    The input is in ascending-rank (descending-score) order, so the first occurrence
    of a key is its highest-scoring prediction. A stable sort by key groups equal
    keys while keeping their input order, so the leading element of each group is the
    winner.
    """
    if keys.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    sorted_keys, sort_index = torch.sort(keys, stable=True)
    is_first_sorted = torch.ones((sorted_keys.numel(),), dtype=torch.bool)
    is_first_sorted[1:] = sorted_keys[1:] != sorted_keys[:-1]
    mask = torch.zeros((keys.numel(),), dtype=torch.bool)
    mask[sort_index[is_first_sorted]] = True
    return mask


def _micro_stats(
    predictions,
    targets,
    iou_threshold: float,
    score_cut: Optional[float] = None,
) -> Dict[str, Any]:
    """Micro precision/recall/F1 at one confidence operating point.

    Every prediction gets a TP/FP verdict by greedy same-label matching in
    descending-score order (mismatched labels are masked to -1 IoU so they can
    never match). Because matching is greedy by score, a prediction's verdict
    does not depend on where a confidence cut lands.
    """
    gt_total = int(sum(len(target['labels']) for target in targets))
    scores_parts = []
    tp_parts = []

    for prediction, target in zip(predictions, targets):
        num_predictions = int(prediction['labels'].numel())
        if num_predictions == 0:
            continue

        order = torch.argsort(prediction['scores'], descending=True, stable=True)
        scores_sorted = prediction['scores'][order]
        tp_flags = torch.zeros((num_predictions,), dtype=torch.bool)

        if target['labels'].numel() > 0:
            pred_labels = prediction['labels'][order]
            pred_boxes = prediction['boxes'][order]
            ious = box_iou(pred_boxes, target['boxes'])
            label_match = pred_labels[:, None] == target['labels'][None, :]
            masked = torch.where(label_match, ious, ious.new_full((), -1.0))
            best_iou, best_gt = torch.max(masked, dim=1)

            valid = best_iou >= iou_threshold
            if bool(valid.any()):
                valid_index = valid.nonzero(as_tuple=False).flatten()
                first_local = _first_occurrence_mask(best_gt[valid_index])
                tp_flags[valid_index[first_local]] = True

        scores_parts.append(scores_sorted)
        tp_parts.append(tp_flags)

    if not scores_parts:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f1_confidence': score_cut}

    scores = torch.cat(scores_parts)
    tp_flags = torch.cat(tp_parts)
    order = torch.argsort(scores, descending=True, stable=True)
    scores = scores[order]
    tp_sorted = tp_flags[order].float()
    if score_cut is None:
        cut = scores.numel()
    else:
        cut = int((scores >= float(score_cut)).sum())

    if cut == 0:
        precision = 0.0
        recall = 0.0
    else:
        tp = float(tp_sorted[:cut].sum().item())
        precision = tp / max(cut, 1)
        recall = tp / max(gt_total, 1)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': _f1(float(precision), float(recall)),
        'f1_confidence': score_cut,
    }


def match_image(prediction, target, iou_threshold: float) -> Dict[str, Any]:
    """Greedily match one image's predictions to its ground truth by IoU (class-agnostic).

    Mirrors the one-gt-per-prediction accounting used by :func:`_confusion_matrix` and the AP
    matching: predictions are claimed in descending score order, each taking its highest-IoU
    still-unclaimed ground-truth box, and a match requires IoU >= ``iou_threshold``. Matching
    ignores class so a located-but-mislabelled box counts as a match with ``class_correct``
    False rather than as both a miss and a false positive.

    Returns ``{'matches': [{'pred_index','gt_index','iou','class_correct'}...],
    'false_positives': [pred_index...], 'misses': [gt_index...]}`` with indices into the
    passed ``prediction``/``target`` arrays.
    """
    pred_boxes = prediction['boxes']
    pred_scores = prediction['scores']
    pred_labels = prediction['labels']
    gt_boxes = target['boxes']
    gt_labels = target['labels']
    num_pred = int(pred_labels.numel())
    num_gt = int(gt_labels.numel())

    gt_claimed = [False] * num_gt
    pred_matched = [False] * num_pred
    matches = []

    if num_pred and num_gt:
        ious = box_iou(pred_boxes, gt_boxes)
        for pred_index in torch.argsort(pred_scores, descending=True, stable=True).tolist():
            available = [gt for gt in range(num_gt) if not gt_claimed[gt]]
            if not available:
                break
            overlaps = ious[pred_index]
            best_gt = max(available, key=lambda gt: float(overlaps[gt]))
            iou = float(overlaps[best_gt])
            if iou < iou_threshold:
                continue
            gt_claimed[best_gt] = True
            pred_matched[pred_index] = True
            matches.append({
                'pred_index': pred_index,
                'gt_index': best_gt,
                'iou': iou,
                'class_correct': int(pred_labels[pred_index]) == int(gt_labels[best_gt]),
            })

    return {
        'matches': matches,
        'false_positives': [index for index in range(num_pred) if not pred_matched[index]],
        'misses': [index for index in range(num_gt) if not gt_claimed[index]],
    }


def _confusion_matrix(
    predictions,
    targets,
    class_ids,
    eval_classes: Optional[Dict[int, str]],
    iou_threshold: float,
    conf_threshold: float,
) -> Dict[str, Any]:
    """Build a class-including confusion matrix at one IoU + confidence threshold.

    Rows are ground-truth classes, columns are predicted classes, with a trailing
    "background" row and column. ``matrix[i][j]`` counts detections whose matched
    ground truth is class ``i`` and whose predicted class is ``j``; the diagonal is
    correct detections and off-diagonal cells are class confusions (a box found but
    mislabelled). The background *row* (index ``len(class_ids)``) counts predictions
    that matched no ground truth — false positives — and the background *column*
    counts ground truths no prediction claimed — misses.

    Each prediction (score >= ``conf_threshold``) is matched greedily by descending
    score to its highest-IoU still-unclaimed ground truth in the same image,
    regardless of class; a match requires IoU >= ``iou_threshold``. This mirrors the
    one-gt-per-prediction accounting used for AP but scores class agreement rather
    than only presence, so a wrong label lands off the diagonal instead of counting
    as both a miss and a false positive.
    """
    index_of = {int(class_id): position for position, class_id in enumerate(class_ids)}
    background = len(class_ids)
    size = background + 1
    matrix = [[0] * size for _ in range(size)]

    for prediction, target in zip(predictions, targets):
        keep = prediction['scores'] >= float(conf_threshold)
        kept = {
            'boxes': prediction['boxes'][keep],
            'scores': prediction['scores'][keep],
            'labels': prediction['labels'][keep],
        }
        gt_labels = target['labels']
        result = match_image(kept, target, iou_threshold)

        for pair in result['matches']:
            truth = index_of.get(int(gt_labels[pair['gt_index']]))
            predicted = index_of.get(int(kept['labels'][pair['pred_index']]))
            if truth is not None and predicted is not None:
                matrix[truth][predicted] += 1

        for pred_index in result['false_positives']:
            predicted = index_of.get(int(kept['labels'][pred_index]))
            if predicted is not None:
                matrix[background][predicted] += 1

        for gt_index in result['misses']:
            truth = index_of.get(int(gt_labels[gt_index]))
            if truth is not None:
                matrix[truth][background] += 1

    return {
        'labels': [_class_name(eval_classes, class_id) or str(class_id) for class_id in class_ids],
        'background_index': background,
        'iou_threshold': float(iou_threshold),
        'conf_threshold': float(conf_threshold),
        'matrix': matrix,
    }


def _micro_precision(predictions, targets, iou_threshold: float) -> float:
    tp, fp, _ = _micro_counts(predictions, targets, iou_threshold)
    return float(tp / max(tp + fp, 1))


def _micro_recall(predictions, targets, iou_threshold: float) -> float:
    tp, _, gt = _micro_counts(predictions, targets, iou_threshold)
    return float(tp / max(gt, 1))


def _micro_counts(predictions, targets, iou_threshold: float) -> tuple[int, int, int]:
    gt_total = int(sum(len(target['labels']) for target in targets))
    total_predictions = 0
    tp = 0

    for prediction, target in zip(predictions, targets):
        num_predictions = int(prediction['labels'].numel())
        total_predictions += num_predictions
        if num_predictions == 0 or target['labels'].numel() == 0:
            continue

        order = torch.argsort(prediction['scores'], descending=True, stable=True)
        pred_labels = prediction['labels'][order]
        pred_boxes = prediction['boxes'][order]

        # For each prediction pick the best same-label ground-truth box. Masking
        # mismatched labels to -1 keeps them below any positive IoU threshold, so
        # they can never be selected as a match.
        ious = box_iou(pred_boxes, target['boxes'])
        label_match = pred_labels[:, None] == target['labels'][None, :]
        masked = torch.where(label_match, ious, ious.new_full((), -1.0))
        best_iou, best_gt = torch.max(masked, dim=1)

        valid = best_iou >= iou_threshold
        if bool(valid.any()):
            valid_index = valid.nonzero(as_tuple=False).flatten()
            first_local = _first_occurrence_mask(best_gt[valid_index])
            tp += int(first_local.sum())

    fp = total_predictions - tp
    return tp, fp, gt_total


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = torch.clamp(boxes1[:, 2] - boxes1[:, 0], min=0) * torch.clamp(boxes1[:, 3] - boxes1[:, 1], min=0)
    area2 = torch.clamp(boxes2[:, 2] - boxes2[:, 0], min=0) * torch.clamp(boxes2[:, 3] - boxes2[:, 1], min=0)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = torch.clamp(bottom_right - top_left, min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / torch.clamp(union, min=1e-12)


def _xywhn_to_xyxy_tensor(boxes: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    x_center, y_center, width, height = boxes.unbind(dim=1)
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return torch.stack([x1, y1, x2, y2], dim=1)


def _average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    if recall.numel() == 0:
        return 0.0

    mrec = torch.cat([recall.new_tensor([0.0]), recall, recall.new_tensor([1.0])])
    mpre = torch.cat([precision.new_tensor([0.0]), precision, precision.new_tensor([0.0])])
    for index in range(mpre.numel() - 1, 0, -1):
        mpre[index - 1] = torch.maximum(mpre[index - 1], mpre[index])
    changing_points = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).flatten()
    ap = torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return float(ap.item())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return float(2 * precision * recall / denominator)
