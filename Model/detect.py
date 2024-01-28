import torch
import yaml
import os
import tqdm
import cv2
import random
import time

from create_model import Model, ModelExtended
from inference_utils import get_bb_from_model_output, get_bb_from_model_output_extended
#from deep_sort_realtime.deepsort_tracker import DeepSort
import xml.etree.ElementTree as ET

class Track():
    def __init__(self, id, cls):
        self.id = id
        self.cls = cls
        self.bboxes = []
        self.frames = []
class DetectTracks():
    def __init__(self, model_config_file, film_path, weights_path, xml_file, every_frame = 1 , verbose = True, visDrone_labels = False, half_precision = True, extended = False):
        
        self.model_config = self.read_yaml(model_config_file)
        self.model_type = self.model_config['model_type']
        self.model_config.pop('model_type')
        self.cls_num = self.model_config['class_number']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.model = self.create_model()
        if torch.cuda.is_available():
            self.model.half().to(device=self.device)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path))
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        self.film_path = film_path
        self.every_frame = every_frame
        self.verbose = verbose
        self.visDrone_labels = visDrone_labels
        self.half_precision = half_precision
        self.tracksList = []
        self.xml_file = xml_file
        self.model_compiled = torch.compile(self.model) 
        self.extended = extended

    def read_yaml(self, config_file):
        with open(config_file) as file:
            yaml_config = yaml.load(file, Loader=yaml.FullLoader)
            
        return yaml_config

    def create_model(self):
        assert self.model_type == "Basic" or self.model_type == "Extended", f"unknown model type :("
            
        if self.model_type == "Basic":
            return Model(**self.model_config)
        elif self.model_type == "Extended":
            return ModelExtended(**self.model_config)

    def detect(self):

        self.model.eval()
        self.model.head.sigm = True
        print(self.film_path)
        #new_film_name = f"output-{self.film_path}"
        cap1 = cv2.VideoCapture(self.film_path)
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #out = cv2.VideoWriter('test2.mp4', fourcc, 30, (1920, 1088))
        frame = 0
        while cap1.isOpened():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            okay1  , frame1 = cap1.read()
            if okay1 == True:
                frame += 1
                if frame%self.every_frame == 0:
                    #frame1 = cv2.resize(frame1, (640, 384))
                    frame1 = cv2.resize(frame1, (1280, 704))
                    img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img)
                    img_tensor = img_tensor.permute(2,0,1)
                    img_tensor = img_tensor.unsqueeze(0)
                    img_tensor = img_tensor/255.

                    #if self.half_precision:
                    #    img_tensor = img_tensor.half()

                    with torch.no_grad():
                        start.record()
                        output = self.model_compiled(img_tensor.half().to(self.device))
                        end.record()
                    torch.cuda.synchronize()
                    fps = 1000./start.elapsed_time(end)
                    print(fps)
                    #for i in output:
                       #print(i[-1].shape)
                        #print(torch.sigmoid(i[-1]))

                    bboxes = get_bb_from_model_output((img_tensor.shape[-2], img_tensor.shape[-1]), output, 0.5, 0.5)
                    for idx, i in enumerate(bboxes["boxes"]):
                        cv2.putText(frame1, str(bboxes["labels"][idx].item()), (int(i[0]+5), int(i[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame1, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255,0,0), 1)

                    #out.write(frame1)        
                    cv2.imshow("vid", frame1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                        #break
            else:
                break
        cap1.release()
        #out.release() 
        cv2.destroyAllWindows()

                    





make_anno = DetectTracks("MHSABDD/create_model.yaml", 'Los Angeles 4K - Night Drive.mp4', "MHSABDD/Custom-DET1_55.pth","annotations.xml", visDrone_labels=False)
make_anno.detect()
# show/DJI_20230914104128_0021_Z.MP4
# show/DJI_20230621083319_0003_Z.MP4