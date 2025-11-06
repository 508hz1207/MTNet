import shutil

import yaml
from addict import Dict
from datetime import datetime
import os
import torch
import torch.nn as nn

class AverageMeter:

    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)

def create_output_dir_and_save_config(config, specified_dir=None):
    if specified_dir:
        output_dir = specified_dir
    else:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    

    config_path = os.path.join(output_dir, "config.yaml")
    
    # Save the config as a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(config.to_dict(), file)
    
    return output_dir
class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def forward(self, target, input):
        eps = 1e-10
        input_ = (input > self.threshold).data.float()
        target_ = (target > self.threshold).data.float()

        intersection = torch.clamp(input_ * target_, 0, 1)
        union = torch.clamp(input_ + target_, 0, 1)

        if torch.mean(intersection).lt(eps):
            return torch.Tensor([0., 0., 0., 0.])
        else:
            acc = torch.mean((input_ == target_).data.float())
            iou = torch.mean(intersection) / torch.mean(union)
            recall = torch.mean(intersection) / torch.mean(target_)
            precision = torch.mean(intersection) / torch.mean(input_)
            F1=(2*precision*recall)/(precision+recall)
            return torch.Tensor([precision, recall, iou, F1])
