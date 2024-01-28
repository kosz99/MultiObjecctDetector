import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import cv2
import yaml

from create_model import Model
from inference_utils import get_bb_from_labels, get_bb_from_model_output
from objDataset_aug import ObjDectDS
from batch_sampler import BatchSampler, RandomSampler, SequentialSampler
torch.set_float32_matmul_precision('high')

def read_yaml(file):
    with open(file) as file:
        yaml_config = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_config

def create_model(model_config):        
    return Model(**model_config)

ds = ObjDectDS("BDD100k", "test", 10)
dataloader = DataLoader(ds, 1, True)

def showAndPred(model_config, model_weight, dataloader, objectness_score=0.5, nms_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = read_yaml(model_config)
    model_config.pop('model_type')
    model = create_model(model_config)
    model = model.to(device=device)
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model.eval()
    model_compiled = torch.compile(model) 

    for _ in range(1):
        for data in tqdm.tqdm(dataloader):
            img = data[0]
            print(img.shape)
            img_to_show = img.squeeze(0).permute(1,2,0).numpy()
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            cv2.imshow("img", img_to_show)
            img_pred = img_to_show.copy()
            img = img.to(device=device)
            with torch.no_grad():
                output = model_compiled(img)
            ######### labels bounding boxes ##########
            bboxes = get_bb_from_labels(data[1])
            if len(bboxes["boxes"])>0:
                for idx, i in enumerate(bboxes["boxes"]):
                    cv2.putText(img_to_show, str(bboxes["labels"][idx].item()), (int(i[0]+5), int(i[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(img_to_show, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255,0,0), 1)
            cv2.imshow("labels", img_to_show)            
            ######### labels bounding boxes ##########
            pred_bboxes = get_bb_from_model_output((img.shape[-2], img.shape[-1]), output, objectness_score=objectness_score, nms_iou_threshold=nms_threshold)
            if len(pred_bboxes["boxes"])>0:
                for idx, i in enumerate(pred_bboxes["boxes"]):
                    cv2.putText(img_pred, str(bboxes["labels"][idx].item()), (int(i[0]+5), int(i[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(img_pred, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255,0,0), 1)
            cv2.imshow("pred_labels", img_pred)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break  
        cv2.destroyAllWindows()

if __name__ == "__main__":
    showAndPred("code-last_training/w_max/create_model.yaml", "code-last_training/w_max/DetPoziom_300.pth", dataloader)