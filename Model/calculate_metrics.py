import torch
import tqdm

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from inference_utils import get_bb_from_labels, get_bb_from_model_output, get_bb_from_labels_extended, get_bb_from_model_output_extended

def calculate_map(model, dataloader, device, obj = 0.25, nms_threshold = 0.7):
    metrics = MeanAveragePrecision(class_metrics=True)
    for data in tqdm.tqdm(dataloader, ncols=50):
        img = data[0]
        with torch.no_grad():
            output = model(img.to(device))
        pred_bboxes = get_bb_from_model_output((img.shape[-2], img.shape[-1]), output, obj, nms_threshold)
        bboxes = get_bb_from_labels(data[1])
        metrics.update([pred_bboxes], [bboxes])

    result = metrics.compute()
    return result

def calculate_map_extended(model, dataloader, device, obj = 0.25, nms_threshold = 0.7):
    metrics = MeanAveragePrecision(class_metrics=True)
    for data in tqdm.tqdm(dataloader, ncols=50):
        img = data[0]
        with torch.no_grad():
            output = model(img.to(device))
        pred_bboxes = get_bb_from_model_output_extended((img.shape[-2], img.shape[-1]), output, obj, nms_threshold)
        bboxes = get_bb_from_labels_extended(data[1])
        metrics.update([pred_bboxes], [bboxes])

    result = metrics.compute()
    return result


