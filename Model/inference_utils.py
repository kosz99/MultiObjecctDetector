import torch
import torchvision

class BBox:
  def __init__(self, x1, y1, x2, y2, cls):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2
    self.cls = cls

  def __str__(self) -> str:
    return f"x1: {self.x1} y1: {self.y1} x2: {self.x2} y2: {self.y2} cls: {self.cls}"

  def __repr__(self) -> str:
    return f"x1: {self.x1} y1: {self.y1} x2: {self.x2} y2: {self.y2} cls: {self.cls}"

  def __eq__(self, other) -> bool:
    return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2 and self.cls == other.cls

  def __hash__(self) -> int:
    return hash((self.x1, self.y1, self.x2, self.y2, self.cls))

class PredictedBbox(BBox):
  def __init__(self, x1, y1, x2, y2, cls, score):
    super().__init__(x1, y1, x2, y2, cls)
    self.score = score
  
  def __str__(self) -> str:
    return f"x1: {self.x1} y1: {self.y1} x2: {self.x2} y2: {self.y2} cls: {self.cls} score: {self.score}"

  def __repr__(self) -> str:
    return f"x1: {self.x1} y1: {self.y1} x2: {self.x2} y2: {self.y2} cls: {self.cls} score: {self.score}"

  def __eq__(self, other):
    return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2 and self.cls == other.cls and self.score == other.score

  def __hash__(self) -> int:
    return hash((self.x1, self.y1, self.x2, self.y2, self.cls, self.score))  

def append_unique(element, tab):
   append = True
   for i in tab:
      if element == i:
         append = False
         break
   if (append == True):
    tab.append(element)
      
def get_bb_from_model_output(img_shape, model_output, objectness_score = 0.5, nms_iou_threshold = 0.5):

    bbox = []
    score = []
    cls = []

    # C5 output shrink 32 times
    C5_cls_output, C5_reg_output, C5_iou_output = model_output[2]
    C5_iou_output = torch.sigmoid(C5_iou_output)
    C5i = C5_iou_output.cpu().numpy()
    C5_iou_output[C5_iou_output>=objectness_score] = 1.0
    C5_iou_output[C5_iou_output<objectness_score] = 0.0   
    C5_reg_output = C5_reg_output*C5_iou_output

    ciou = C5_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (32*x)
        _y = (32*y)
        bbox.append(torch.tensor([max(0, _x - C5_reg_output[0,0,y,x].item()), max(0, _y - C5_reg_output[0,2,y,x].item()), min(img_shape[1], _x + C5_reg_output[0,1,y,x].item()), min(img_shape[0], _y + C5_reg_output[0,3,y,x].item())]))           
        score.append(C5i[0,0,y,x])
        cls.append(torch.argmax(C5_cls_output[0,:,y,x]).item())
    
    #C4 output shrink 16 times
    C4_cls_output, C4_reg_output, C4_iou_output = model_output[1]
    C4_iou_output = torch.sigmoid(C4_iou_output)
    C4i = C4_iou_output.cpu().numpy()
    C4_iou_output[C4_iou_output>=objectness_score] = 1.0
    C4_iou_output[C4_iou_output<objectness_score] = 0.0   
    C4_reg_output = C4_reg_output*C4_iou_output

    ciou = C4_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (16*x) 
        _y = (16*y) 
        bbox.append(torch.tensor([max(0, _x - C4_reg_output[0,0,y,x].item()), max(0, _y - C4_reg_output[0,2,y,x].item()), min(img_shape[1], _x + C4_reg_output[0,1,y,x].item()), min(img_shape[0], _y + C4_reg_output[0,3,y,x].item())]))
        score.append(C4i[0,0,y,x])
        cls.append(torch.argmax(C4_cls_output[0,:,y,x]).item())  
    
    #C3 output shrink 8 times
    C3_cls_output, C3_reg_output, C3_iou_output = model_output[0]
    C3_iou_output = torch.sigmoid(C3_iou_output)
    C3i = C3_iou_output.cpu().numpy()
    C3_iou_output[C3_iou_output>=objectness_score] = 1.0
    C3_iou_output[C3_iou_output<objectness_score] = 0.0   
    C3_reg_output = C3_reg_output*C3_iou_output

    ciou = C3_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (8*x) 
        _y = (8*y) 
        bbox.append(torch.tensor([max(0, _x - C3_reg_output[0,0,y,x].item()), max(0, _y - C3_reg_output[0,2,y,x].item()), min(img_shape[1], _x + C3_reg_output[0,1,y,x].item()), min(img_shape[0], _y + C3_reg_output[0,3,y,x].item())]))
        score.append(C3i[0,0,y,x])
        cls.append(torch.argmax(C3_cls_output[0,:,y,x]).item())

    if len(bbox)>0:
        Bbox_tesnor = torch.stack(bbox, dim=0)
        Score_tensor = torch.tensor(score)
        nms = torchvision.ops.nms(Bbox_tesnor, Score_tensor.type(torch.FloatTensor), nms_iou_threshold)

        boxes = []
        scores = []
        bb_cls = []
        for n in nms:
            bb = [bbox[n][0].item(), bbox[n][1].item(), bbox[n][2].item(), bbox[n][3].item()]
            boxes.append(bb)
            scores.append(score[n])
            bb_cls.append(cls[n])

            

        return {"boxes" : torch.tensor(boxes), "scores" : torch.tensor(scores), "labels" : torch.tensor(bb_cls)}    
    else:
        return {"boxes" : torch.tensor([]), "scores" : torch.tensor([]), "labels" : torch.tensor([])}

def get_bb_from_model_output_without_cls(model_output, objectness_score = 0.5, nms_iou_threshold = 0.5):

    bbox = []
    score = []
    cls = []

    # C5 output shrink 32 times
    C5_cls_output, C5_reg_output, C5_iou_output = model_output[2]
    C5i = C5_iou_output.cpu().numpy()
    C5_iou_output[C5_iou_output>=objectness_score] = 1.0
    C5_iou_output[C5_iou_output<objectness_score] = 0.0   
    C5_reg_output = C5_reg_output*C5_iou_output

    ciou = C5_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (32*x)
        _y = (32*y)
        bbox.append(torch.tensor([_x - C5_reg_output[0,0,y,x].item(), _y - C5_reg_output[0,2,y,x].item(), _x + C5_reg_output[0,1,y,x].item(), _y + C5_reg_output[0,3,y,x].item()]))           
        #score.append(C5i[0,0,y,x])
        score.append(1.0)
        cls.append(torch.argmax(C5_cls_output[0,:,y,x]).item())
    
    #C4 output shrink 16 times
    C4_cls_output, C4_reg_output, C4_iou_output = model_output[1]
    C4i = C4_iou_output.cpu().numpy()
    C4_iou_output[C4_iou_output>=objectness_score] = 1.0
    C4_iou_output[C4_iou_output<objectness_score] = 0.0   
    C4_reg_output = C4_reg_output*C4_iou_output

    ciou = C4_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (16*x) 
        _y = (16*y) 
        bbox.append(torch.tensor([_x - C4_reg_output[0,0,y,x].item(), _y - C4_reg_output[0,2,y,x].item(), _x + C4_reg_output[0,1,y,x].item(), _y + C4_reg_output[0,3,y,x].item()]))
        #score.append(C4i[0,0,y,x])
        score.append(1.0)
        cls.append(torch.argmax(C4_cls_output[0,:,y,x]).item())  
    
    #C3 output shrink 8 times
    C3_cls_output, C3_reg_output, C3_iou_output = model_output[0]
    C3i = C3_iou_output.cpu().numpy()
    C3_iou_output[C3_iou_output>=objectness_score] = 1.0
    C3_iou_output[C3_iou_output<objectness_score] = 0.0   
    C3_reg_output = C3_reg_output*C3_iou_output

    ciou = C3_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (8*x) 
        _y = (8*y) 
        bbox.append(torch.tensor([_x - C3_reg_output[0,0,y,x].item(), _y - C3_reg_output[0,2,y,x].item(), _x + C3_reg_output[0,1,y,x].item(), _y + C3_reg_output[0,3,y,x].item()]))
        #score.append(C3i[0,0,y,x])
        score.append(1.0)
        cls.append(torch.argmax(C3_cls_output[0,:,y,x]).item())

    if len(bbox)>0:
        Bbox_tesnor = torch.stack(bbox, dim=0)
        Score_tensor = torch.tensor(score)
        nms = torchvision.ops.nms(Bbox_tesnor, Score_tensor.type(torch.FloatTensor), nms_iou_threshold)

        boxes = []
        scores = []
        bb_cls = []
        for n in nms:
            bb = [bbox[n][0].item(), bbox[n][1].item(), bbox[n][2].item(), bbox[n][3].item()]
            boxes.append(bb)
            scores.append(score[n])
            bb_cls.append(0)

        return {"boxes" : torch.tensor(boxes), "scores" : torch.tensor(scores), "labels" : torch.tensor(bb_cls)}    
    else:
        return {"boxes" : torch.tensor([]), "scores" : torch.tensor([]), "labels" : torch.tensor([])}

def get_bb_from_labels(data):

    C3_cls, C4_cls, C5_cls, C3_iou, C4_iou, C5_iou, C3_reg, C4_reg, C5_reg = data

    bboxes = []
    new_tab = []

    #C5_cls, C5_reg, C5_iou
    ciou = C5_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (32*x)
        _y = (32*y)

        append_unique(BBox(_x - C5_reg[0,0,y,x].item(),
                          _y - C5_reg[0,2,y,x].item(),
                          _x + C5_reg[0,1,y,x].item(),
                          _y + C5_reg[0,3,y,x].item(),
                          torch.argmax(C5_cls[0,:,y,x]).item()), bboxes)

    #C4_cls, C4_reg, C4_iou
    ciou = C4_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (16*x)
        _y = (16*y)

        append_unique(BBox(_x - C4_reg[0,0,y,x].item(),
                          _y - C4_reg[0,2,y,x].item(),
                          _x + C4_reg[0,1,y,x].item(),
                          _y + C4_reg[0,3,y,x].item(),
                          torch.argmax(C4_cls[0,:,y,x]).item()), bboxes)

    #C3_cls, C3_reg, C3_iou
    ciou = C3_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (8*x)
        _y = (8*y)

        append_unique(BBox(_x - C3_reg[0,0,y,x].item(),
                          _y - C3_reg[0,2,y,x].item(),
                          _x + C3_reg[0,1,y,x].item(),
                          _y + C3_reg[0,3,y,x].item(),
                          torch.argmax(C3_cls[0,:,y,x]).item()), bboxes)
    boxes = []
    bb_cls = []

    for i in bboxes:
       boxes.append([i.x1, i.y1, i.x2, i.y2])
       bb_cls.append(i.cls)
    
    if len(boxes)>0:
       return {"boxes" : torch.tensor(boxes), "labels" : torch.tensor(bb_cls)}
    
    else:
       return {"boxes" : torch.tensor([]), "labels" : torch.tensor([])}


    

def get_bb_from_labels_without_cls(data):

    C3_cls, C4_cls, C5_cls, C3_iou, C4_iou, C5_iou, C3_reg, C4_reg, C5_reg = data

    bboxes = []
    new_tab = []

    C5_cls, C5_reg, C5_iou
    ciou = C5_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (32*x)
        _y = (32*y)

        append_unique(BBox(_x - C5_reg[0,0,y,x].item(),
                          _y - C5_reg[0,2,y,x].item(),
                          _x + C5_reg[0,1,y,x].item(),
                          _y + C5_reg[0,3,y,x].item(),
                          torch.argmax(C5_cls[0,:,y,x]).item()), bboxes)

    C4_cls, C4_reg, C4_iou
    ciou = C4_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (16*x)
        _y = (16*y)

        append_unique(BBox(_x - C4_reg[0,0,y,x].item(),
                          _y - C4_reg[0,2,y,x].item(),
                          _x + C4_reg[0,1,y,x].item(),
                          _y + C4_reg[0,3,y,x].item(),
                          torch.argmax(C4_cls[0,:,y,x]).item()), bboxes)

    C3_cls, C3_reg, C3_iou
    ciou = C3_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (8*x)
        _y = (8*y)

        append_unique(BBox(_x - C3_reg[0,0,y,x].item(),
                          _y - C3_reg[0,2,y,x].item(),
                          _x + C3_reg[0,1,y,x].item(),
                          _y + C3_reg[0,3,y,x].item(),
                          torch.argmax(C3_cls[0,:,y,x]).item()), bboxes)
    boxes = []
    bb_cls = []

    for i in bboxes:
       boxes.append([i.x1, i.y1, i.x2, i.y2])
       bb_cls.append(0)
    
    if len(boxes)>0:
       return {"boxes" : torch.tensor(boxes), "labels" : torch.tensor(bb_cls)}
    
    else:
       return {"boxes" : torch.tensor([]), "labels" : torch.tensor([])}





def get_bb_from_model_output_extended(img_shape, model_output, objectness_score = 0.5, nms_iou_threshold = 0.5):

    bbox = []
    score = []
    cls = []

    # C6 output shrink 64 times
    C6_cls_output, C6_reg_output, C6_iou_output = model_output[3]
    C6_iou_output = torch.sigmoid(C6_iou_output)
    C6i = C6_iou_output.cpu().numpy()
    C6_iou_output[C6_iou_output>=objectness_score] = 1.0
    C6_iou_output[C6_iou_output<objectness_score] = 0.0   
    C6_reg_output = C6_reg_output*C6_iou_output

    ciou = C6_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (64*x)
        _y = (64*y)
        bbox.append(torch.tensor([max(0, _x - C6_reg_output[0,0,y,x].item()), max(0, _y - C6_reg_output[0,2,y,x].item()), min(img_shape[1], _x + C6_reg_output[0,1,y,x].item()), min(img_shape[0], _y + C6_reg_output[0,3,y,x].item())]))           
        score.append(C6i[0,0,y,x])
        #score.append(1.0)
        cls.append(torch.argmax(C6_cls_output[0,:,y,x]).item())


    # C5 output shrink 32 times
    C5_cls_output, C5_reg_output, C5_iou_output = model_output[2]
    C5_iou_output = torch.sigmoid(C5_iou_output)
    C5i = C5_iou_output.cpu().numpy()
    C5_iou_output[C5_iou_output>=objectness_score] = 1.0
    C5_iou_output[C5_iou_output<objectness_score] = 0.0   
    C5_reg_output = C5_reg_output*C5_iou_output

    ciou = C5_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (32*x)
        _y = (32*y)
        bbox.append(torch.tensor([max(0, _x - C5_reg_output[0,0,y,x].item()), max(0, _y - C5_reg_output[0,2,y,x].item()), min(img_shape[1], _x + C5_reg_output[0,1,y,x].item()), min(img_shape[0], _y + C5_reg_output[0,3,y,x].item())]))           
        score.append(C5i[0,0,y,x])
        #score.append(1.0)
        cls.append(torch.argmax(C5_cls_output[0,:,y,x]).item())
    
    #C4 output shrink 16 times
    C4_cls_output, C4_reg_output, C4_iou_output = model_output[1]
    C4_iou_output = torch.sigmoid(C4_iou_output)
    C4i = C4_iou_output.cpu().numpy()
    C4_iou_output[C4_iou_output>=objectness_score] = 1.0
    C4_iou_output[C4_iou_output<objectness_score] = 0.0   
    C4_reg_output = C4_reg_output*C4_iou_output

    ciou = C4_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (16*x) 
        _y = (16*y) 
        bbox.append(torch.tensor([max(0, _x - C4_reg_output[0,0,y,x].item()), max(0, _y - C4_reg_output[0,2,y,x].item()), min(img_shape[1], _x + C4_reg_output[0,1,y,x].item()), min(img_shape[0], _y + C4_reg_output[0,3,y,x].item())]))
        score.append(C4i[0,0,y,x])
        #score.append(1.0)
        cls.append(torch.argmax(C4_cls_output[0,:,y,x]).item())  
    
    #C3 output shrink 8 times
    C3_cls_output, C3_reg_output, C3_iou_output = model_output[0]
    C3_iou_output = torch.sigmoid(C3_iou_output)
    C3i = C3_iou_output.cpu().numpy()
    C3_iou_output[C3_iou_output>=objectness_score] = 1.0
    C3_iou_output[C3_iou_output<objectness_score] = 0.0   
    C3_reg_output = C3_reg_output*C3_iou_output

    ciou = C3_iou_output.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (8*x) 
        _y = (8*y) 
        bbox.append(torch.tensor([max(0, _x - C3_reg_output[0,0,y,x].item()), max(0, _y - C3_reg_output[0,2,y,x].item()), min(img_shape[1], _x + C3_reg_output[0,1,y,x].item()), min(img_shape[0], _y + C3_reg_output[0,3,y,x].item())]))
        score.append(C3i[0,0,y,x])
        #score.append(1.0)
        cls.append(torch.argmax(C3_cls_output[0,:,y,x]).item())

    if len(bbox)>0:
        Bbox_tesnor = torch.stack(bbox, dim=0)
        Score_tensor = torch.tensor(score)
        nms = torchvision.ops.nms(Bbox_tesnor, Score_tensor.type(torch.FloatTensor), nms_iou_threshold)

        boxes = []
        scores = []
        bb_cls = []
        for n in nms:
            bb = [bbox[n][0].item(), bbox[n][1].item(), bbox[n][2].item(), bbox[n][3].item()]
            boxes.append(bb)
            scores.append(score[n])
            bb_cls.append(cls[n])

            

        return {"boxes" : torch.tensor(boxes), "scores" : torch.tensor(scores), "labels" : torch.tensor(bb_cls)}    
    else:
        return {"boxes" : torch.tensor([]), "scores" : torch.tensor([]), "labels" : torch.tensor([])}


def get_bb_from_labels_extended(data):

    C3_cls, C4_cls, C5_cls, C6_cls, C3_iou, C4_iou, C5_iou, C6_iou, C3_reg, C4_reg, C5_reg, C6_reg = data

    bboxes = []
    new_tab = []

    #C6_cls, C6_reg, C6_iou
    ciou = C6_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (64*x)
        _y = (64*y)

        append_unique(BBox(_x - C6_reg[0,0,y,x].item(),
                          _y - C6_reg[0,2,y,x].item(),
                          _x + C6_reg[0,1,y,x].item(),
                          _y + C6_reg[0,3,y,x].item(),
                          torch.argmax(C6_cls[0,:,y,x]).item()), bboxes)


    #C5_cls, C5_reg, C5_iou
    ciou = C5_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (32*x)
        _y = (32*y)

        append_unique(BBox(_x - C5_reg[0,0,y,x].item(),
                          _y - C5_reg[0,2,y,x].item(),
                          _x + C5_reg[0,1,y,x].item(),
                          _y + C5_reg[0,3,y,x].item(),
                          torch.argmax(C5_cls[0,:,y,x]).item()), bboxes)

    #C4_cls, C4_reg, C4_iou
    ciou = C4_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (16*x)
        _y = (16*y)

        append_unique(BBox(_x - C4_reg[0,0,y,x].item(),
                          _y - C4_reg[0,2,y,x].item(),
                          _x + C4_reg[0,1,y,x].item(),
                          _y + C4_reg[0,3,y,x].item(),
                          torch.argmax(C4_cls[0,:,y,x]).item()), bboxes)

    #C3_cls, C3_reg, C3_iou
    ciou = C3_iou.to_sparse().indices()
    for i in range(len(ciou[0])):
        _,_,y,x = ciou[:,i]
        _x = (8*x)
        _y = (8*y)

        append_unique(BBox(_x - C3_reg[0,0,y,x].item(),
                          _y - C3_reg[0,2,y,x].item(),
                          _x + C3_reg[0,1,y,x].item(),
                          _y + C3_reg[0,3,y,x].item(),
                          torch.argmax(C3_cls[0,:,y,x]).item()), bboxes)
    boxes = []
    bb_cls = []

    for i in bboxes:
       boxes.append([i.x1, i.y1, i.x2, i.y2])
       bb_cls.append(i.cls)
    
    if len(boxes)>0:
       return {"boxes" : torch.tensor(boxes), "labels" : torch.tensor(bb_cls)}    
    else:
       return {"boxes" : torch.tensor([]), "labels" : torch.tensor([])}
    





'''
tab = []
b1 = BBox(782.0, 71.0, 832.0, 112.0, 2)
b2 = BBox(782.0, 71.0, 832.0, 112.0, 2)
tab.append(b1)
tab.append(b2)
print(tab)
print(set(tab))
'''