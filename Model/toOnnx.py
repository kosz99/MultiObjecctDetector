import torch
import yaml
import os
import tqdm
import cv2
import random
import torch.onnx

from create_model import Model, ModelExtended


class Convert2onnx():
    def __init__(self, model_config_file, weights_path):
        self.model_config = self.read_yaml(model_config_file)
        self.model_type = self.model_config['model_type']
        self.model_config.pop('model_type')
        self.model = self.create_model()
        self.model.to(device="cpu")
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

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

    def convert(self):
        batch_size = 1
        x = torch.randn(batch_size, 3, 736, 1280, requires_grad=False)
        torch_out = self.model(x)
        torch.onnx.export(self.model,               # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        "test.onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})




Convert = Convert2onnx("neww_w/create_model.yaml", "neww_w/BDD100K-2_100.pth")
Convert.convert()