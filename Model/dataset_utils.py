import cv2
import torch


class Bbox:
    def __init__(self, x1, x2, y1, y2, cls):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.cls = cls
    
    def __str__(self):
        return f"X1: {self.x1} Y1: {self.y1} X2: {self.x2} Y2: {self.y2} cls: {self.cls}"
    
def get_img(img_path):
    #print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print("-------------")
    #print(img)
    return img

def resize_bbox(anno_path, img_shape, new_image_shape, cls_names_list, cls2idx):
    tabBbox = []
    with open(anno_path) as f:
        txtBbox = f.readlines()
    for txt in txtBbox:
        txt = txt.split(" ")
        if txt[0] in cls_names_list:
                
            x1 = int(float(txt[1]))
            y1 = int(float(txt[2]))
            x2 = x1 + int(float(txt[3]))
            y2 = y1 + int(float(txt[4]))
            cls = cls2idx[txt[0]]

            

            x1_new = int((x1/img_shape[0])*new_image_shape[0])
            x2_new = int((x2/img_shape[0])*new_image_shape[0])

            y1_new = int((y1/img_shape[1])*new_image_shape[1])
            y2_new = int((y2/img_shape[1])*new_image_shape[1])

            tabBbox.append(Bbox(x1_new, x2_new, y1_new, y2_new, cls))

    return tabBbox    

def multiple_resize_bbox(anno_pathes, old_sizes, new_image_shape, cls_names_list, cls2idx):
    tabBbox = []
    for id, a_path in enumerate(anno_pathes):
        with open(a_path) as f:
            txtBbox = f.readlines()
        for txt in txtBbox:
            txt = txt.split(" ")
            if txt[0] in cls_names_list:
                x1 = int(float(txt[1]))
                y1 = int(float(txt[2]))
                x2 = x1 + int(float(txt[3]))
                y2 = y1 + int(float(txt[4]))
                cls = cls2idx[txt[0]]
                

                x1_new = int((x1/old_sizes[id][0])*new_image_shape[0])
                x2_new = int((x2/old_sizes[id][0])*new_image_shape[0])

                y1_new = int((y1/old_sizes[id][1])*new_image_shape[1])
                y2_new = int((y2/old_sizes[id][1])*new_image_shape[1])

                tabBbox.append(Bbox(x1_new, x2_new, y1_new, y2_new, cls))
    
    return tabBbox

def bias_multiple_resize_bbox(anno_pathes, old_sizes, new_image_shape, bias, cls_names_list, cls2idx):
    tabBbox = []
    for id, a_path in enumerate(anno_pathes):
        with open(a_path) as f:
            txtBbox = f.readlines()
        for txt in txtBbox:
            txt = txt.split(" ")
            if txt[0] in cls_names_list:
                x1 = int(float(txt[1]))
                y1 = int(float(txt[2]))
                x2 = x1 + int(float(txt[3]))
                y2 = y1 + int(float(txt[4]))
                cls = cls2idx[txt[0]]

                x1_new = int((x1/old_sizes[id][0])*new_image_shape[0]) + bias[id][0]
                x2_new = int((x2/old_sizes[id][0])*new_image_shape[0]) + bias[id][0]

                y1_new = int((y1/old_sizes[id][1])*new_image_shape[1]) + bias[id][1]
                y2_new = int((y2/old_sizes[id][1])*new_image_shape[1]) + bias[id][1]

                tabBbox.append(Bbox(x1_new, x2_new, y1_new, y2_new, cls))
    
    return tabBbox

def create_mask(tabBox, new_image_shape, num_classes, small_size, medium_size):
    C5_mask_shape = (new_image_shape[1]//32, new_image_shape[0]//32)
    C4_mask_shape = (new_image_shape[1]//16, new_image_shape[0]//16)
    C3_mask_shape = (new_image_shape[1]//8, new_image_shape[0]//8)

    C5_mask_cls = torch.zeros(num_classes, C5_mask_shape[0], C5_mask_shape[1])
    C5_mask_iou = torch.zeros(1, C5_mask_shape[0], C5_mask_shape[1])
    C5_mask_reg = torch.zeros(4, C5_mask_shape[0], C5_mask_shape[1])
    
    C4_mask_cls = torch.zeros(num_classes, C4_mask_shape[0], C4_mask_shape[1])
    C4_mask_iou = torch.zeros(1, C4_mask_shape[0], C4_mask_shape[1])
    C4_mask_reg = torch.zeros(4, C4_mask_shape[0], C4_mask_shape[1])

    C3_mask_cls = torch.zeros(num_classes, C3_mask_shape[0], C3_mask_shape[1])
    C3_mask_iou = torch.zeros(1, C3_mask_shape[0], C3_mask_shape[1])            
    C3_mask_reg = torch.zeros(4, C3_mask_shape[0], C3_mask_shape[1])

    for t in tabBox:
        center = int((t.x2 - t.x1)/2) + t.x1, int((t.y2 - t.y1)/2) + t.y1
        l,r,top,b = center[0] - t.x1, t.x2 - center[0], center[1] - t.y1, t.y2 - center[1]
        #print("Przetwarzam_teraz: ", t)
        #m = max(l,r, top, b)
        width = int((t.x2 - t.x1))
        height = int((t.y2 - t.y1))
        m = max(height, width)
        minimum = min(height, width)
        if minimum>10 and m<small_size: #C3
            x = center[0]//8
            y = center[1]//8

            for x_ in range(x-1, x+2):
                for y_ in range(y-1, y+2):
                    if ((x_<C3_mask_shape[1]) and ( y_<C3_mask_shape[0])):
                        xx = 8*x_
                        yy = 8*y_
                        if ((xx > t.x1) and (xx < t.x2) and (yy > t.y1) and (yy < t.y2)):
                                        
                            C3_mask_reg[0,y_,x_] = xx - t.x1
                            C3_mask_reg[1,y_,x_] = t.x2 - xx
                            C3_mask_reg[2,y_,x_] = yy - t.y1
                            C3_mask_reg[3,y_,x_] = t.y2 - yy

                            C3_mask_cls[t.cls, y_, x_] = 1.0

                            C3_mask_iou[0, y_, x_] = 1.0

        if m<medium_size and m>=small_size: #C4
            x = center[0]//16
            y = center[1]//16

            for x_ in range(x-1, x+2):
                for y_ in range(y-1, y+2):
                    if ((x_<C4_mask_shape[1]) and ( y_<C4_mask_shape[0])):
                        xx = 16*x_
                        yy = 16*y_
                        if ((xx > t.x1) and (xx < t.x2) and (yy > t.y1) and (yy < t.y2)):
                                    
                            C4_mask_reg[0,y_,x_] = xx - t.x1
                            C4_mask_reg[1,y_,x_] = t.x2 - xx
                            C4_mask_reg[2,y_,x_] = yy - t.y1
                            C4_mask_reg[3,y_,x_] = t.y2 - yy

                            C4_mask_cls[t.cls, y_, x_] = 1.0

                            C4_mask_iou[0, y_, x_] = 1.0

        if m>=medium_size: #C5
            x = center[0]//32
            y = center[1]//32

            for x_ in range(x-1, x+2):
                for y_ in range(y-1, y+2):
                    if ((x_<C5_mask_shape[1]) and ( y_<C5_mask_shape[0])):
                        xx = 32*x_
                        yy = 32*y_
                        if ((xx > t.x1) and (xx < t.x2) and (yy > t.y1) and (yy < t.y2)):
                                    
                            C5_mask_reg[0,y_,x_] = xx - t.x1
                            C5_mask_reg[1,y_,x_] = t.x2 - xx
                            C5_mask_reg[2,y_,x_] = yy - t.y1
                            C5_mask_reg[3,y_,x_] = t.y2 - yy

                            C5_mask_cls[t.cls, y_, x_] = 1.0
                            
                            C5_mask_iou[0, y_, x_] = 1.0
    
    return C3_mask_cls, C4_mask_cls, C5_mask_cls, C3_mask_iou, C4_mask_iou, C5_mask_iou, C3_mask_reg, C4_mask_reg, C5_mask_reg


def create_maskExtended(tabBox, new_image_shape, num_classes, small_size, medium_size, large_size):
    C6_mask_shape = (new_image_shape[1]//64, new_image_shape[0]//64)
    C5_mask_shape = (new_image_shape[1]//32, new_image_shape[0]//32)
    C4_mask_shape = (new_image_shape[1]//16, new_image_shape[0]//16)
    C3_mask_shape = (new_image_shape[1]//8, new_image_shape[0]//8)

    C6_mask_cls = torch.zeros(num_classes, C6_mask_shape[0], C6_mask_shape[1])
    C6_mask_iou = torch.zeros(1, C6_mask_shape[0], C6_mask_shape[1])
    C6_mask_reg = torch.zeros(4, C6_mask_shape[0], C6_mask_shape[1])

    C5_mask_cls = torch.zeros(num_classes, C5_mask_shape[0], C5_mask_shape[1])
    C5_mask_iou = torch.zeros(1, C5_mask_shape[0], C5_mask_shape[1])
    C5_mask_reg = torch.zeros(4, C5_mask_shape[0], C5_mask_shape[1])
    
    C4_mask_cls = torch.zeros(num_classes, C4_mask_shape[0], C4_mask_shape[1])
    C4_mask_iou = torch.zeros(1, C4_mask_shape[0], C4_mask_shape[1])
    C4_mask_reg = torch.zeros(4, C4_mask_shape[0], C4_mask_shape[1])

    C3_mask_cls = torch.zeros(num_classes, C3_mask_shape[0], C3_mask_shape[1])
    C3_mask_iou = torch.zeros(1, C3_mask_shape[0], C3_mask_shape[1])            
    C3_mask_reg = torch.zeros(4, C3_mask_shape[0], C3_mask_shape[1])

    for t in tabBox:
        center = int((t.x2 - t.x1)/2) + t.x1, int((t.y2 - t.y1)/2) + t.y1
        l,r,top,b = center[0] - t.x1, t.x2 - center[0], center[1] - t.y1, t.y2 - center[1]
        #print("Przetwarzam_teraz: ", t)
        #m = max(l,r, top, b)
        width = int((t.x2 - t.x1))
        height = int((t.y2 - t.y1))
        m = max(height, width)
        minimum = min(height, width)
        if minimum>25 and m<small_size: #C3
            x = center[0]//8
            y = center[1]//8

            for x_ in range(x-1, x+2):
                for y_ in range(y-1, y+2):
                    if ((x_<C3_mask_shape[1]) and ( y_<C3_mask_shape[0])):
                        xx = 8*x_
                        yy = 8*y_
                        if ((xx > t.x1) and (xx < t.x2) and (yy > t.y1) and (yy < t.y2)):
                                        
                            C3_mask_reg[0,y_,x_] = xx - t.x1
                            C3_mask_reg[1,y_,x_] = t.x2 - xx
                            C3_mask_reg[2,y_,x_] = yy - t.y1
                            C3_mask_reg[3,y_,x_] = t.y2 - yy

                            C3_mask_cls[t.cls, y_, x_] = 1.0

                            C3_mask_iou[0, y_, x_] = 1.0

        if m<medium_size and m>=small_size: #C4
            x = center[0]//16
            y = center[1]//16

            for x_ in range(x-1, x+2):
                for y_ in range(y-1, y+2):
                    if ((x_<C4_mask_shape[1]) and ( y_<C4_mask_shape[0])):
                        xx = 16*x_
                        yy = 16*y_
                        if ((xx > t.x1) and (xx < t.x2) and (yy > t.y1) and (yy < t.y2)):
                                    
                            C4_mask_reg[0,y_,x_] = xx - t.x1
                            C4_mask_reg[1,y_,x_] = t.x2 - xx
                            C4_mask_reg[2,y_,x_] = yy - t.y1
                            C4_mask_reg[3,y_,x_] = t.y2 - yy

                            C4_mask_cls[t.cls, y_, x_] = 1.0

                            C4_mask_iou[0, y_, x_] = 1.0

        if m<large_size and m>=medium_size: #C5
            x = center[0]//32
            y = center[1]//32

            for x_ in range(x-1, x+2):
                for y_ in range(y-1, y+2):
                    if ((x_<C5_mask_shape[1]) and ( y_<C5_mask_shape[0])):
                        xx = 32*x_
                        yy = 32*y_
                        if ((xx > t.x1) and (xx < t.x2) and (yy > t.y1) and (yy < t.y2)):
                                    
                            C5_mask_reg[0,y_,x_] = xx - t.x1
                            C5_mask_reg[1,y_,x_] = t.x2 - xx
                            C5_mask_reg[2,y_,x_] = yy - t.y1
                            C5_mask_reg[3,y_,x_] = t.y2 - yy

                            C5_mask_cls[t.cls, y_, x_] = 1.0
                            
                            C5_mask_iou[0, y_, x_] = 1.0

        if m>=large_size: #C6
            x = center[0]//64
            y = center[1]//64

            for x_ in range(x-1, x+2):
                for y_ in range(y-1, y+2):
                    if ((x_<C6_mask_shape[1]) and ( y_<C6_mask_shape[0])):
                        xx = 64*x_
                        yy = 64*y_
                        if ((xx > t.x1) and (xx < t.x2) and (yy > t.y1) and (yy < t.y2)):
                                    
                            C6_mask_reg[0,y_,x_] = xx - t.x1
                            C6_mask_reg[1,y_,x_] = t.x2 - xx
                            C6_mask_reg[2,y_,x_] = yy - t.y1
                            C6_mask_reg[3,y_,x_] = t.y2 - yy

                            C6_mask_cls[t.cls, y_, x_] = 1.0
                            
                            C6_mask_iou[0, y_, x_] = 1.0
    
    return C3_mask_cls, C4_mask_cls, C5_mask_cls, C6_mask_cls, C3_mask_iou, C4_mask_iou, C5_mask_iou, C6_mask_iou, C3_mask_reg, C4_mask_reg, C5_mask_reg, C6_mask_reg