import os
import numpy as np
import imageio
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse 
"""
为了可视化更改一下dataset
"""

class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        if self.is_train:
            train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        else:
            train_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                          train_file_list[:data_len]]
        self.train_img_path = [os.path.join(self.root, 'images', train_file) for train_file in
                          train_file_list[:data_len]]
        self.train_img_name = [train_file for train_file in
                          train_file_list[:data_len]]
        if self.is_train:
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        else:
            self.train_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        self.train_img_name = [x for x in train_file_list[:data_len]]

    def __getitem__(self, index):
        img, target = self.train_img[index], self.train_label[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img,self.train_img_path[index],self.train_img_name[index], target

    def __len__(self):
            return len(self.train_label)

class getFeatureMap(torch.nn.Module):
    def __init__(self,args):
        super(getFeatureMap, self).__init__()
        #self.net = torchvision.models.resnet152(pretrained = True)
        if args.net == 'resnet18':
            self.net = torchvision.models.resnet18(pretrained = True)
        elif args.net == 'resnet34':
            self.net = torchvision.models.resnet34(pretrained = True)
        elif args.net == 'resnet50':
            self.net = torchvision.models.resnet50(pretrained = True)
        elif args.net == 'resnet101':
            self.net = torchvision.models.resnet101(pretrained = True)
        elif args.net == 'resnet152':
            self.net = torchvision.models.resnet152(pretrained = True)
        else:
            raise ValueError('please choose correct net!')
    
    def forward(self,x):
        conv1 = self.net.conv1(x)
        bn1 = self.net.bn1(conv1)
        relu = self.net.relu(bn1)
        maxpool = self.net.maxpool(relu)

        layer1 = self.net.layer1(maxpool)
        layer2 = self.net.layer2(layer1)
        layer3 = self.net.layer3(layer2)
        layer4 = self.net.layer4(layer3)

        avgpool = self.net.avgpool(layer4)
        y = torch.flatten(avgpool, 1)
        y = self.net.fc(y)

        feature_dict = {
            'conv1':conv1,
            'bn1':bn1,
            'relu':relu,
            'maxpool':maxpool,
            'layer1':layer1,
            'layer2':layer2,
            'layer3':layer3,
            'layer4':layer4
        }
        return feature_dict

def getDataLoader(root,train):

    imageTransform = transforms.Compose([transforms.Resize((448, 448), Image.BILINEAR),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    myData = CUB(root=root, is_train=train,transform=imageTransform)

    myLoader = torch.utils.data.DataLoader(myData, batch_size=1, shuffle=False,
                                                           num_workers=0, pin_memory=True)
    return myLoader

def drawFeature(features_dict,imgOrignPath,imgName,input_root,output_root):
    # batchsize 设置为1
    imgOrignPath = imgOrignPath[0]
    imgName = imgName[0]
    for conv_name, features in features_dict.items():
        heat = features.data.cpu().numpy()  # 将tensor格式的feature map转为numpy格式
        heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
        heatmaps = np.maximum(heat, 0)  # heatmap与0比较

        img = cv2.imread(imgOrignPath)
        heatmapSum = np.sum(heatmaps, axis=0)  # 多通道时，取均值
        heatmapSum/=np.max(heatmapSum)
        heatmapSum = cv2.resize(heatmapSum, (img.shape[1], img.shape[0]))
        heatmapSum = np.uint8(255 * heatmapSum)
        heatmapSum = cv2.applyColorMap(heatmapSum, cv2.COLORMAP_JET)
        heat_img = cv2.addWeighted(img, 1, heatmapSum, 0.4, 0)
        savePath = os.path.join(imgOrignPath.replace(input_root,output_root))
        savePath1 = savePath.replace(os.path.basename(savePath).split('.')[-1],conv_name + "_hotmap" + '.' + os.path.basename(savePath).split('.')[-1])
        savePath2 = savePath.replace(os.path.basename(savePath).split('.')[-1],conv_name + "_hotimg" + '.' + os.path.basename(savePath).split('.')[-1])
        if os.path.isdir(os.path.dirname(savePath)) == False:
            os.makedirs(os.path.dirname(savePath))
        cv2.imwrite(savePath1,heatmapSum)
        cv2.imwrite(savePath2,heat_img)


def main(root,train,args):
    net = getFeatureMap(args)
    Imageloader = getDataLoader(root,train)
    with torch.no_grad():
        net.eval()
        for imgs,imgOrignPath,imgName,_ in tqdm(Imageloader,desc='Processing', unit='item',position=0,total=len(Imageloader)):
            features = net(imgs)
            drawFeature(features, imgOrignPath, imgName,root,args.outdir)


"""
模型最大准确率83.24
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser value')
    parser.add_argument('--net', type=str,default='resnet101',choices=['resnet18', 'resnet34','resnet50', 'resnet101','resnet152'])
    parser.add_argument('--outdir',type=str)
    root = '/data/kb/tanyuanyong/TransFG-master/data/CUB_200_2011'
    args = parser.parse_args()
    main(root,True,args)


