import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import cv2

from inference_utils import get_bb_from_labels
from Dataset import ObjDectDS
from batch_sampler import BatchSampler, RandomSampler, SequentialSampler


ds = ObjDectDS("BDD100k", "train", 10)
dataloader = DataLoader(ds,
                        batch_sampler=BatchSampler(
                            RandomSampler(ds),
                            1,
                            False,
                            1,
                            [(1270, 704)])
                            )

def show_ds(dataloader):
    for _ in range(1):
        for data in dataloader:
            img = data[0]
            print(img.shape)
            img_to_show = img.squeeze(0).permute(1,2,0).numpy()
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            cv2.imshow("img", img_to_show)
            ######### labels bounding boxes ########## bboxes["labels"][idx].item()
            bboxes = get_bb_from_labels(data[1])
            if len(bboxes["boxes"])>0:
                print(len(bboxes["boxes"]))
                for idx, i in enumerate(bboxes["boxes"]):
                    cv2.putText(img_to_show, str(bboxes["labels"][idx].item()), (int(i[0]+5), int(i[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(img_to_show, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 1)
            cv2.imshow("labels", img_to_show)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break  
        cv2.destroyAllWindows()

if __name__=="__main__":
    show_ds(dataloader)