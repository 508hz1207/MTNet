1. Required libraries:
(1) an Nvidia GPU with latest CUDA and driver.
(2) the latest pytorch
(3) reformer-pytorch\opencv-python\matplotlib\scipy\thop\matplotlib\wandb

2. Dataset links:
(1)CHN6-CUG: https://github.com/CUG-URS/CHN6-CUG-Roads-Dataset
(2)City-scale:https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H

3. Main code files:
(1)Training: train.py
(2)Inference:inferencer.py 

4.Metric Evaluation:
go to run CHN6_CUG/metric_evaluation.py

5. Other Notes:
The PWRoad , a multi-source data fusion dataset designed for general road extraction scenarios, will be released later. In addition, a new dataset named INRoad, focusing on typical urban complex overpass scenarios and also constructed based on multi-source data fusion, will be introduced. Meanwhile, the complete code will be open-sourced as well.

