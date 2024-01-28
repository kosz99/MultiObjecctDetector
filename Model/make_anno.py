import torch
import yaml
import os
import tqdm
import cv2

from create_model import Model, ModelExtended
from inference_utils import get_bb_from_model_output_extended

class CreateAnno():
    def __init__(self, model_config_file, film_path, weights_path, every_frame = 50, img_size = (1280, 704), verbose = True, visDrone_labels = False, half_precision = False):
        
        self.model_config = self.read_yaml(model_config_file)
        self.model_type = self.model_config['model_type']
        self.model_config.pop('model_type')
        self.cls_num = self.model_config['class_number']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.model = self.create_model()
        if torch.cuda.is_available():
            self.model.to(device=self.device)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path))
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        self.film_path = film_path
        self.every_frame = every_frame
        self.img_size = img_size
        self.verbose = verbose
        self.visDrone_labels = visDrone_labels
        self.half_precision = half_precision

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

    def create_anno(self):

        for i in os.listdir(os.path.join("GenerateAnno","result", "imgs")):
            os.remove(os.path.join("GenerateAnno", "result", "imgs", i))
        
        for i in os.listdir(os.path.join("GenerateAnno", "result", "obj_train_data")):
            os.remove(os.path.join("GenerateAnno", "result", "obj_train_data", i))

        self.model.eval()
        self.model.head.sigm = True
        cap1 = cv2.VideoCapture(self.film_path)
        frame = 0
        while cap1.isOpened():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            okay1  , frame1 = cap1.read()
            if okay1 == True:
                frame += 1
                if frame%self.every_frame == 0:
                    new_img = cv2.resize(frame1, self.img_size)
                    f = open(os.path.join("GenerateAnno", "result", "obj_train_data", f"{frame}.txt"), "w")
                    cv2.imwrite(os.path.join("GenerateAnno", "result", "imgs", f"{frame}.jpg"), new_img)
                    img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img)
                    img_tensor = img_tensor.permute(2,0,1)
                    img_tensor = img_tensor.unsqueeze(0)
                    img_tensor = img_tensor/255.

                    with torch.no_grad():
                        start.record()
                        output = self.model(img_tensor.to(self.device))
                        end.record()
                    torch.cuda.synchronize()
                    fps = 1000./start.elapsed_time(end)
                    print(fps)
                
                    bboxes = get_bb_from_model_output_extended((img_tensor.shape[-2], img_tensor.shape[-1]), output, 0.25, 0.7)
                    #print(bboxes)
                    for idx, anno in enumerate(bboxes['boxes']):

                        f.write(f"{int(bboxes['labels'][idx].item())} {(((anno[2]-anno[0])/2)+anno[0])/self.img_size[0]} {(((anno[3]-anno[1])/2)+anno[1])/self.img_size[1]} {(anno[2]-anno[0])/self.img_size[0]} {(anno[3]-anno[1])/self.img_size[1]}")
                        f.write('\n')
                    f.close()
                    for idx, i in enumerate(bboxes["boxes"]):
                        cv2.putText(new_img, str(bboxes["labels"][idx].item()), (int(i[0]+5), int(i[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(new_img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255,0,0), 1)
                            
                    cv2.imshow("vid", new_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        cap1.release()
        cv2.destroyAllWindows()

make_anno = CreateAnno("Models/MHSABDD/create_model.yaml", "vid/Zurich Switzerland 4k HDR - Sunset Drive - Driving Downtown.mp4", 'Models/MHSABDD/Custom-DET1_55.pth', visDrone_labels=True)
make_anno.create_anno()