import torch
import yaml
import os
import tqdm
from torch.utils.data import DataLoader
from batch_sampler import BatchSampler, RandomSampler, SequentialSampler
from create_model import Model, ModelExtended
from utils import get_optimizer, get_lrScheduler
from loss import Loss, LossExtended
from calculate_metrics import calculate_map, calculate_map_extended
from Dataset import ObjDectDS
from clearml import Task, Logger

torch.backends.cudnn.benchmark = True
  
class Trainer:
    def __init__(self, model_config_file, training_config_file):
        
        self.model_config = self.read_yaml(model_config_file)
        self.model_type = self.model_config['model_type']
        self.model_config.pop('model_type')
        self.cls_num = self.model_config['class_number']

        self.device = "cuda:0"
        self.model = self.create_model()
        self.model.to(device=self.device)

        self.training_config = self.read_yaml(training_config_file)
        self.warmup_epochs = self.training_config['warmup_epochs']
        self.last_epochs_without_aug = self.training_config['last_epochs_without_aug']
        self.check_gpu(self.training_config['gpu_num'])
        if self.training_config['gpu_num']>1:
            self.m = self.model
            self.model = torch.nn.DataParallel(self.m, device_ids=[i for i in range(self.training_config['gpu_num'])])

        self.check_workers(self.training_config['num_workers'])
        
        self.optimizer = get_optimizer(self.model, **self.training_config['optim_setup'])
 
        self.loss = self.create_loss(**self.training_config['loss_setup'])

        if self.training_config['gradScaler']:
            self.scaler = torch.cuda.amp.GradScaler()
            

        if self.model_type == "Basic":

            self.train_dataset = ObjDectDS(self.training_config['dataset_setup']['dataset_folder'],
                                        self.training_config['dataset_setup']['train_dataset'],
                                        self.cls_num,
                                        self.training_config['dataset_setup']['train_mosaic_prob'],
                                        self.training_config['dataset_setup']['train_mixup_prob'],
                                        self.training_config['dataset_setup']['train_colorJitter_prob'],
                                        self.training_config['dataset_setup']['train_mosaic_colorJitter_prob'],
                                        self.training_config['dataset_setup']['train_mixup_colorJitter_prob'],
                                        self.training_config['dataset_setup']['train_gauss_noise_prob'],
                                        self.training_config['dataset_setup']['train_mosaic_gauss_noise_prob'],
                                        self.training_config['dataset_setup']['train_mixup_gauss_noise_prob'],
                                        self.training_config['dataset_setup']['small_size'],
                                        self.training_config['dataset_setup']['medium_size'],
                                        self.training_config['dataset_setup']['large_size'],
                                        False)
            
            self.test_dataset = ObjDectDS(self.training_config['dataset_setup']['dataset_folder'],
                                        self.training_config['dataset_setup']['val_dataset'],
                                        self.cls_num,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        self.training_config['dataset_setup']['small_size'],
                                        self.training_config['dataset_setup']['medium_size'],
                                        self.training_config['dataset_setup']['large_size'],
                                        False)
            
            self.train_dataset_withoutAug = ObjDectDS(self.training_config['dataset_setup']['dataset_folder'],
                                        self.training_config['dataset_setup']['train_dataset'],
                                        self.cls_num,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        self.training_config['dataset_setup']['small_size'],
                                        self.training_config['dataset_setup']['medium_size'],
                                        self.training_config['dataset_setup']['large_size'],
                                        False)
        
        elif self.model_type == "Extended":
            self.train_dataset = ObjDectDS(self.training_config['dataset_setup']['dataset_folder'],
                                        self.training_config['dataset_setup']['train_dataset'],
                                        self.cls_num,
                                        self.training_config['dataset_setup']['train_mosaic_prob'],
                                        self.training_config['dataset_setup']['train_mixup_prob'],
                                        self.training_config['dataset_setup']['train_colorJitter_prob'],
                                        self.training_config['dataset_setup']['train_mosaic_colorJitter_prob'],
                                        self.training_config['dataset_setup']['train_mixup_colorJitter_prob'],
                                        self.training_config['dataset_setup']['train_gauss_noise_prob'],
                                        self.training_config['dataset_setup']['train_mosaic_gauss_noise_prob'],
                                        self.training_config['dataset_setup']['train_mixup_gauss_noise_prob'],
                                        self.training_config['dataset_setup']['small_size'],
                                        self.training_config['dataset_setup']['medium_size'],
                                        self.training_config['dataset_setup']['large_size'],
                                        True)
            
            self.test_dataset = ObjDectDS(self.training_config['dataset_setup']['dataset_folder'],
                                        self.training_config['dataset_setup']['val_dataset'],
                                        self.cls_num,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        self.training_config['dataset_setup']['small_size'],
                                        self.training_config['dataset_setup']['medium_size'],
                                        self.training_config['dataset_setup']['large_size'],
                                        True)
            
            self.train_dataset_withoutAug = ObjDectDS(self.training_config['dataset_setup']['dataset_folder'],
                                        self.training_config['dataset_setup']['train_dataset'],
                                        self.cls_num,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        self.training_config['dataset_setup']['small_size'],
                                        self.training_config['dataset_setup']['medium_size'],
                                        self.training_config['dataset_setup']['large_size'],
                                        True)
            
            
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_sampler=BatchSampler(
                                               RandomSampler(self.train_dataset),
                                               self.training_config['dataloader_setup']['train_batch_size'],
                                               False,
                                               self.training_config['dataloader_setup']['train_multiscale_step'],
                                               self.training_config['dataloader_setup']['train_img_size']),
                                           num_workers=self.training_config['num_workers'],
                                           pin_memory=self.training_config['pin_memory'] 
                                           )   

        self.train_dataloader_withoutAug = DataLoader(self.train_dataset,
                                           batch_sampler=BatchSampler(
                                               RandomSampler(self.train_dataset),
                                               self.training_config['dataloader_setup']['train_batch_size'],
                                               False,
                                               self.training_config['dataloader_setup']['train_multiscale_step'],
                                               self.training_config['dataloader_setup']['train_img_size']),
                                           num_workers=self.training_config['num_workers'],
                                           pin_memory=self.training_config['pin_memory'] 
                                           ) 

        self.test_dataloader = DataLoader(self.test_dataset,
                                           batch_sampler=BatchSampler(
                                               SequentialSampler(self.test_dataset),
                                               32,
                                               False,
                                               0,
                                               self.training_config['dataloader_setup']['val_img_size']),
                                           num_workers=self.training_config['num_workers'],
                                           pin_memory=self.training_config['pin_memory'] 
                                           ) 

        self.map_dataloader = DataLoader(self.test_dataset,
                                           batch_sampler=BatchSampler(
                                               SequentialSampler(self.test_dataset),
                                               1,
                                               False,
                                               0,
                                               self.training_config['dataloader_setup']['val_img_size']),
                                           num_workers=self.training_config['num_workers'],
                                           pin_memory=self.training_config['pin_memory'] 
                                           ) 

        if self.training_config['LR_setup']['LR']:
            self.scheduler_name = self.training_config['LR_setup']['name']
            self.main_scheduler = get_lrScheduler(self.scheduler_name,
                                                  self.optimizer,
                                                  self.training_config['LR_setup']['max_lr'],
                                                  len(self.train_dataloader),
                                                  self.training_config['epochs'] - self.training_config['warmup_epochs'],
                                                  self.training_config['LR_setup']['T_max'],
                                                  self.training_config['LR_setup']['eta_min'],
                                                  False)    
            #print(self.main_scheduler)                                                       

        '''
        self.metrcics_dataloader = DataLoader(self.test_dataset,
                                           batch_sampler=BatchSampler(
                                           SequentialSampler(self.test_dataset),
                                           1,
                                           False,
                                           0,
                                           self.training_config['dataloader_setup']['val_img_size']
                                           ),
                                           num_workers=8,
                                           pin_memory=True)
        '''

    def create_loss(self, reg_loss = "DIOU", obj_loss = "Focal", cls_loss = "BCE", alpha = 1, beta = 1, gamma = 1):
        if self.model_type == "Basic":
            return Loss(reg_loss, obj_loss, cls_loss, alpha, beta, gamma)
        
        elif self.model_type == "Extended":
            return LossExtended(reg_loss, obj_loss, cls_loss, alpha, beta, gamma)    

    def read_yaml(self,config_file):
        with open(config_file) as file:
            yaml_config = yaml.load(file, Loader=yaml.FullLoader)
        
        return yaml_config
        
    def create_model(self):
        assert self.model_type == "Basic" or self.model_type == "Extended", f"unknown model type :("
        
        if self.model_type == "Basic":
            return Model(**self.model_config)
        elif self.model_type == "Extended":
            return ModelExtended(**self.model_config)

    def check_gpu(self, gpu_num):
        assert gpu_num <= torch.cuda.device_count(), f"too little gpus :("

    def check_workers(self, workers_num):
        assert workers_num <= os.cpu_count(), f"too much workers :("
    
    def warmup(self):

        warmup_optimizer = get_optimizer(self.model, **self.training_config['optim_setup'])
        scheduler = torch.optim.lr_scheduler.LinearLR(warmup_optimizer, self.training_config['initial_lr']/self.training_config['optim_setup']['learning_rate'], total_iters=self.training_config['warmup_epochs'], verbose=True)


        for epoch in range(self.training_config['warmup_epochs']):
            self.model.train()
            train_loss = 0.0
            cls_loss = 0.0
            reg_loss = 0.0
            iou_loss = 0.0
            n_pos = 0.0

            for data in tqdm.tqdm(self.train_dataloader, ncols=50):
                if self.model_type == "Basic":
                    img, label = data
                    C3_cls, C4_cls, C5_cls, C3_iou, C4_iou, C5_iou, C3_reg, C4_reg, C5_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device))

                if self.model_type == "Extended":
                    img, label = data
                    C3_cls, C4_cls, C5_cls, C6_cls, C3_iou, C4_iou, C5_iou, C6_iou, C3_reg, C4_reg, C5_reg, C6_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C6_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C6_iou.to(device=self.device),C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device), C6_reg.to(device=self.device))


                img = img.to(device=self.device)
                self.optimizer.zero_grad()
                if self.training_config['gradScaler']:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        output = self.model(img)
                        l, l_cls, l_iou, l_reg, npos = self.loss(output, labels)
                    
                    self.scaler.scale(l).backward()
                    self.scaler.step(warmup_optimizer)
                    self.scaler.update()
                else:
                    output = self.model(img)
                    l, l_cls, l_iou, l_reg, npos = self.loss(output, labels)
                    l.backward()
                    warmup_optimizer.step()

                train_loss += l.item()
                cls_loss += l_cls.item()
                reg_loss += l_reg.item()
                iou_loss += l_iou.item()
                n_pos += npos.item()
            
            scheduler.step()

            print("(Warmup) Epoch: ", epoch+1, "Train Loss: ", train_loss/len(self.train_dataloader), "Cls Loss: ", cls_loss/n_pos, "Reg Loss: ", reg_loss/n_pos, "Iou Loss: ", iou_loss/len(self.train_dataloader))
            Logger.current_logger().report_scalar("Loss", "train", train_loss/len(self.train_dataloader), epoch+1)
            Logger.current_logger().report_scalar("Cls_Loss", "train", cls_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("Reg_Loss", "train", reg_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("IOU_Loss", "train", iou_loss/len(self.train_dataloader), epoch+1)  


            self.model.eval()
            test_loss = 0.0
            cls_loss = 0.0
            reg_loss = 0.0
            iou_loss = 0.0
            n_pos = 0.0            

            for data in tqdm.tqdm(self.test_dataloader, ncols=50):
                if self.model_type == "Basic":
                    img, label = data
                    C3_cls, C4_cls, C5_cls, C3_iou, C4_iou, C5_iou, C3_reg, C4_reg, C5_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device))
                
                elif self.model_type == "Extended":
                    img, label = data
                    C3_cls, C4_cls, C5_cls, C6_cls, C3_iou, C4_iou, C5_iou, C6_iou, C3_reg, C4_reg, C5_reg, C6_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C6_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C6_iou.to(device=self.device), C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device), C6_reg.to(device=self.device))

                img = img.to(device=self.device)

                with torch.no_grad():
                    output = self.model(img)
                
                l, l_cls, l_iou, l_reg, npos = self.loss(output, labels)

                test_loss += l.item()
                cls_loss += l_cls.item()
                reg_loss += l_reg.item()
                iou_loss += l_iou.item()
                n_pos += npos.item()
            
            print("(Warmup) Epoch: ", epoch+1, "Test Loss: ", test_loss/len(self.test_dataloader), "Cls Loss: ", cls_loss/n_pos, "Reg Loss: ", reg_loss/n_pos, "Iou Loss: ", iou_loss/len(self.test_dataloader))
            Logger.current_logger().report_scalar("Loss", "test", test_loss/len(self.test_dataloader), epoch+1)
            Logger.current_logger().report_scalar("Cls_Loss", "test", cls_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("Reg_Loss", "test", reg_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("IOU_Loss", "test", iou_loss/len(self.test_dataloader), epoch+1) 



    def train(self):
        task = Task.init(project_name="TestTrening", task_name=self.training_config['training_name'])

        #Optional warmup
        start_epoch_num = 0
        if self.training_config['warmup']:
            self.warmup()
            start_epoch_num = self.training_config['warmup_epochs']
        
        # Train loop
        idx = 0
        for epoch in range(start_epoch_num, self.training_config['epochs']):
            self.model.train()
            train_loss = 0.0
            cls_loss = 0.0
            reg_loss = 0.0
            iou_loss = 0.0
            n_pos = 0.0

            if self.training_config['epochs'] - epoch > self.last_epochs_without_aug:
                dataloader = self.train_dataloader
            else:
                dataloader = self.train_dataloader_withoutAug             
            for data in tqdm.tqdm(dataloader, ncols=50):
                if self.model_type == "Basic":
                    img, label = data
                  

                    C3_cls, C4_cls, C5_cls, C3_iou, C4_iou, C5_iou, C3_reg, C4_reg, C5_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device))

                if self.model_type == "Extended":
                    img, label = data

                    C3_cls, C4_cls, C5_cls, C6_cls, C3_iou, C4_iou, C5_iou, C6_iou, C3_reg, C4_reg, C5_reg, C6_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C6_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C6_iou.to(device=self.device),C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device), C6_reg.to(device=self.device))


                img = img.to(device=self.device)
                self.optimizer.zero_grad()
                if self.training_config['gradScaler']:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        output = self.model(img)
                        l, l_cls, l_iou, l_reg, npos = self.loss(output, labels)
                    
                    self.scaler.scale(l).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(img)
                    l, l_cls, l_iou, l_reg, npos = self.loss(output, labels)
                    l.backward()
                    self.optimizer.step()
                
                #scheduler one cycle
                if self.training_config['LR_setup']['LR'] and self.scheduler_name == "OneCycle":
                    self.main_scheduler.step()
                    Logger.current_logger().report_scalar("LR", "lr", self.main_scheduler.get_last_lr()[0], idx)
                    

                train_loss += l.item()
                cls_loss += l_cls.item()
                reg_loss += l_reg.item()
                iou_loss += l_iou.item()
                n_pos += npos.item()

            #scheduler cos
            if self.training_config['LR_setup']['LR'] and self.scheduler_name == "Cosine":
                self.main_scheduler.step()
                Logger.current_logger().report_scalar("LR", "lr", self.main_scheduler.get_last_lr()[0], idx)
            print("Epoch: ", epoch+1, "Train Loss: ", train_loss/len(self.train_dataloader), "Cls Loss: ", cls_loss/n_pos, "Reg Loss: ", reg_loss/n_pos, "Iou Loss: ", iou_loss/len(self.train_dataloader))
            Logger.current_logger().report_scalar("Loss", "train", train_loss/len(self.train_dataloader), epoch+1)
            Logger.current_logger().report_scalar("Cls_Loss", "train", cls_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("Reg_Loss", "train", reg_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("IOU_Loss", "train", iou_loss/len(self.train_dataloader), epoch+1)  

            # Eval testdataset
            self.model.eval()
            test_loss = 0.0
            cls_loss = 0.0
            reg_loss = 0.0
            iou_loss = 0.0
            n_pos = 0.0            

            for data in tqdm.tqdm(self.test_dataloader, ncols=50):
                if self.model_type == "Basic":
                    img, label = data
                    
                    C3_cls, C4_cls, C5_cls, C3_iou, C4_iou, C5_iou, C3_reg, C4_reg, C5_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device))
                
                elif self.model_type == "Extended":
                    img, label = data
                    C3_cls, C4_cls, C5_cls, C6_cls, C3_iou, C4_iou, C5_iou, C6_iou, C3_reg, C4_reg, C5_reg, C6_reg = label
                    labels = (C3_cls.to(device=self.device), C4_cls.to(device=self.device), C5_cls.to(device=self.device), C6_cls.to(device=self.device), C3_iou.to(device=self.device), C4_iou.to(device=self.device), C5_iou.to(device=self.device), C6_iou.to(device=self.device), C3_reg.to(device=self.device), C4_reg.to(device=self.device), C5_reg.to(device=self.device), C6_reg.to(device=self.device))

                img = img.to(device=self.device)

                with torch.no_grad():
                    output = self.model(img)
                
                l, l_cls, l_iou, l_reg, npos = self.loss(output, labels)

                test_loss += l.item()
                cls_loss += l_cls.item()
                reg_loss += l_reg.item()
                iou_loss += l_iou.item()
                n_pos += npos.item()
            
            print("Epoch: ", epoch+1, "Test Loss: ", test_loss/len(self.test_dataloader), "Cls Loss: ", cls_loss/n_pos, "Reg Loss: ", reg_loss/n_pos, "Iou Loss: ", iou_loss/len(self.test_dataloader))
            Logger.current_logger().report_scalar("Loss", "test", test_loss/len(self.test_dataloader), epoch+1)
            Logger.current_logger().report_scalar("Cls_Loss", "test", cls_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("Reg_Loss", "test", reg_loss/n_pos, epoch+1)
            Logger.current_logger().report_scalar("IOU_Loss", "test", iou_loss/len(self.test_dataloader), epoch+1)                
            
            if (epoch+1)%self.training_config['save_weights']['calculate_map_every_epoch'] == 0 or (epoch+1)%self.training_config['save_weights']['save_after_every_epochs'] == 0:
                    
                if self.model_type == "Basic":
                    metrics = calculate_map(self.model, self.map_dataloader, self.device)
                elif self.model_type == "Extended":
                    metrics = calculate_map_extended(self.model, self.map_dataloader, self.device)    
            #    print(metrics)
                Logger.current_logger().report_scalar("mAP", "mAP", metrics['map'].item(), epoch+1)
                Logger.current_logger().report_scalar("mAP", "mAP@50", metrics['map_50'].item(), epoch+1)
                Logger.current_logger().report_scalar("mAP", "mAP@75", metrics['map_75'].item(), epoch+1)
                
                for cls_idx, map_cls in enumerate(metrics['map_per_class']):
                    Logger.current_logger().report_scalar("mAP per class", f"class_{cls_idx}", map_cls.item(), epoch+1)

                for cls_idx, mar_cls in enumerate(metrics['mar_100_per_class']):
                    Logger.current_logger().report_scalar("mAR per class", f"class_{cls_idx}", mar_cls.item(), epoch+1)
            

            

            if (epoch+1)%self.training_config['save_weights']['save_after_every_epochs'] == 0:
                name = f"{self.training_config['save_weights']['dir_name']}/{self.training_config['training_name']}_{epoch+1}.pth"
                if self.training_config['gpu_num']>1:
                    torch.save(self.model.module.state_dict(), name)
                else:
                    torch.save(self.model.state_dict(), name)

 
if __name__=="__main__":

    trainer = Trainer("Model/create_model.yaml", "Model/config_training.yaml")
    trainer.train()
