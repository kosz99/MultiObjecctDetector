import torch
import yaml
import cv2
import time

from create_model import Model, ModelExtended
from inference_utils import get_bb_from_model_output, get_bb_from_model_output_extended

class Infer:
    def __init__(self, model_config_file, film_path, weights_path, infer_every_frame = 1, half_precision = False, objectness_score = 0.5, nms_iou_threshold = 0.5):
        
        self.model_config = self.read_yaml(model_config_file)
        self.model_type = self.model_config['model_type']
        self.model_config.pop('model_type')
        self.cls_num = self.model_config['class_number']

        self.half_precision = half_precision
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.model = self.create_model()
        if torch.cuda.is_available():
            if self.half_precision:
                self.model.half().to(device=self.device)
            else:
                self.model.half().to(device=self.device)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path))
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        print("Inference device: ", self.device)
        self.film_path = film_path
        self.every_frame = infer_every_frame
        self.objectness_score = objectness_score
        self.nms_iou_threshold = nms_iou_threshold
        self.compute = True
        if torch.cuda.is_available():
            self.model = torch.compile(self.model) 

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

    def infer(self):
        self.model.eval()
        print("Film path:", self.film_path)
        cap1 = cv2.VideoCapture(self.film_path)
        frame = 0
        while cap1.isOpened():
            okay1, frame1 = cap1.read()
            if okay1 == True:
                frame += 1
                if frame%self.every_frame == 0:
                    frame1 = cv2.resize(frame1, (1280, 704))
                    if self.compute:
                        img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                        img_tensor = torch.from_numpy(img)
                        img_tensor = img_tensor.permute(2,0,1)
                        img_tensor = img_tensor.unsqueeze(0)
                        img_tensor = img_tensor/255.
                        if self.half_precision:
                            img_tensor = img_tensor.half()
                        start = time.time()
                        with torch.no_grad():
                            output = self.model(img_tensor.to(self.device))
                        bboxes = get_bb_from_model_output((img_tensor.shape[-2], img_tensor.shape[-1]), output, self.objectness_score, self.nms_iou_threshold)
                        end = time.time()
                        fps = 1./(end-start)
                        print("FPS: ", fps)
                        for idx, i in enumerate(bboxes["boxes"]):
                            cv2.putText(frame1, str(bboxes["labels"][idx].item()), (int(i[0]+5), int(i[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(frame1, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255,0,0), 1)       
                    cv2.imshow("vid", frame1)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('c'):
                        self.compute = not self.compute
                    if key == ord('p'):
                        cv2.waitKey(-1)
            else:
                break
        cap1.release()
        cv2.destroyAllWindows()

                    



if __name__=="__main__":
    make_anno = Infer("MultiScaleModel/create_model.yaml", "GX010013.MP4", "MultiScaleModel/Multiscale_15.pth")
    make_anno.infer()
