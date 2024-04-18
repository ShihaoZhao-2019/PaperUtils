from PIL import Image
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import cv2
import hiddenlayer as h

MyConvNet = torchvision.models.vgg11(True)

vis_graph = h.build_graph(MyConvNet, torch.zeros([1 ,3,448,448]))
    
vis_graph.theme = h.graph.THEMES["blue"].copy()

if os.path.exists('/output/') != False:
    os.makedirs('/output/')
vis_graph.save("./output/vgg11",format='png')
