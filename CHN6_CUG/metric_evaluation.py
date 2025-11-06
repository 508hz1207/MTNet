
import torch
import numpy as np
import cv2
import os
import glob
import operator
from collections import Counter
import segmentation_models_pytorch as smp
import tqdm
import shutil
import pandas as pd

class caculate_iou():
    def __init__(self):
        self.value=0
        self.num=0
        self.total_value=0
        self.avg=0

    def updata(self,value,n=1):
        self.value=value
        self.num+=n
        self.total_value+=value*n
        self.avg=self.total_value/self.num

def caculate_index(name_num):
    index=[]
    temp=name_num[0]
    index.append(temp)
    for num in name_num[1:]:
        temp+=num
        index.append(temp)
    return index

def cat_image_iou():
    path1=r"/home/whu/wzj/third/MT-RoadNet-surface/CHN6_CUG/result/b_inferencer_result/surface"

    ious=caculate_iou()
    f1_scores=caculate_iou()
    precision=caculate_iou()
    recall=caculate_iou()
    preds=glob.glob(os.path.join(path1,"*pre.png"))
    gts=glob.glob(os.path.join(path1,"*gt.png"))
    for i, index in tqdm.tqdm(enumerate(preds), desc="进程", total=80):
        pred_image = cv2.imread(preds[i], cv2.IMREAD_GRAYSCALE)
        pred_image = np.where(pred_image == 255, 1, 0)
        gt_path=preds[i].replace("pre","gt")
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_image = np.where(gt_image == 255, 1, 0)
        cat_pred_image=np.where(pred_image==0,0,255)
        cat_gt_image=np.where(gt_image==0,0,255)
        cat_pred_image=torch.as_tensor(cat_pred_image)[None,None,:,:]
        cat_gt_image = torch.as_tensor(cat_gt_image)[None,None,:,:]
        batch_stats = smp.metrics.get_stats(
            cat_pred_image,
            cat_gt_image,
            mode='binary',
            threshold=0.5,
        )
        num_images=1
        batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
        batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
        batch_precision=smp.metrics.precision(*batch_stats, reduction="micro-imagewise")
        batch_recall=smp.metrics.recall(*batch_stats, reduction="micro-imagewise")
        ious.updata(batch_iou.item(), num_images)
        f1_scores.updata(batch_f1.item(), num_images)
        precision.updata(batch_precision.item(), num_images)
        recall.updata(batch_recall.item(), num_images)
        # print("ious,%.4f"%ious.avg, "|", "f1_scores:%.4f"%f1_scores.avg, "|", "precision:%.4f"% precision.avg, "|", "recall%.4f"%recall.avg,"name:",single_names[i])
        print("precision:%.4f"% precision.avg, "|", "recall%.4f"%recall.avg,"|", "f1_scores:%.4f"%f1_scores.avg,"|", "ious,%.4f"%ious.avg,  "name:",os.path.basename( preds[i]).split("_")[0])

    print("ious,  %.4f" % ious.avg," f1_scores,  %.4f" % f1_scores.avg," precision,  %.4f" % precision.avg,
          " recall,  %.4f" % recall.avg)

if __name__=="__main__":
    cat_image_iou()
    print()