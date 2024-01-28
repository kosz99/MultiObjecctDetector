import torch
import yaml
import os
import tqdm
import cv2

from create_model import Model, ModelExtended
from inference_utils import get_bb_from_model_output

colors ={
    0 : (255, 0, 0), # niebieski human
    1: (153, 51, 255), # rozowy bicycle
    2: (0, 0, 255), # czerwony car
    3: (204, 0, 102), # fitulowy truck
    4: (204, 0, 102), # pomaranczowy bus

}

def read_yaml(config_file):
    with open(config_file) as file:
        yaml_config = yaml.load(file, Loader=yaml.FullLoader)
            
    return yaml_config

yaml_name = "dysk/code/create_model.yaml"
model_config = read_yaml(yaml_name)
model_config.pop('model_type')
model = Model(**model_config)
model = model.cuda(0)
model.load_state_dict(torch.load("dysk/code/weights/ARGOVERSE_5.pth"))
model.eval()
model.head.sigm = True

cap1 = cv2.VideoCapture('dysk/filmy/Warsaw - Evening Drive Through Downtown - 4K.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('dysk/filmy/output-Warsaw - Evening Drive Through Downtown - 4K.mp4', fourcc, 30, (1280, 736))

every_frame = 1
frame = 0
while cap1.isOpened():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    okay1  , frame1 = cap1.read()
    if okay1 == True:
        frame += 1
        if frame%every_frame == 0:
            new_img = cv2.resize(frame1, (1280, 736))
            img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.permute(2,0,1)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor/255.  

            with torch.no_grad():
                start.record()
                output = model(img_tensor.cuda(0))
                end.record()

            torch.cuda.synchronize()
            val = 1000./start.elapsed_time(end)
            print("FPS:", val)
            bboxes = get_bb_from_model_output(output)

            for idx, i in enumerate(bboxes["boxes"]):
                #cv2.rectangle(new_img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), colors[bboxes["cls"][idx].item()], 1)
                cv2.rectangle(new_img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 1)
            out.write(new_img)

out.release()