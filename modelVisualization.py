from PIL import Image
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import cv2
import hiddenlayer as h
   
MyConvNet = torchvision.models.vgg16(True)
 
vis_graph = h.build_graph(MyConvNet, torch.zeros([1 ,3,448,448]))
    
vis_graph.theme = h.graph.THEMES["blue"].copy()
    
vis_graph.save("./output/vgg16.png")
