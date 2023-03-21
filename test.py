n_downsampling = 2
ngf = 64
import torch
import cv2
import torchvision.transforms as transforms
import torch.nn as nn
# x = torch.randn((4,1024,8,8)) # BCHW or NCHW
# layer1 = nn.ConvTranspose2d(1024,1024,kernel_size=1)
# y1 = layer1(x) # [4, 1024, 8, 8]
# # print(y1.shape,'l1')
#
# layer2 = nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2,padding=1,output_padding=1)
# y2 = layer2(y1)
# # print(y2.shape,'l2')  # 4, 512, 16, 16]
#
# layer3 = nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1)
# y3 = layer3(y2)
# # print(y3.shape,'l3')  #[4, 512,  32, 32]
#
# layer4 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,output_padding=1)
# y4 = layer4(y3)
# print(y4.shape,'l4') # [4, 256,  64, 64]

# for i in range(n_downsampling):
#     mult = 2 ** i
#     print("ngf*mult",ngf*mult,"ngf*mult*2",ngf*mult*2)
#     if i == 1:
#         content_encoder1 =
#     else:
#         content_encoder2 =
#
# yolo = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
#
# pic1 = cv2.imread('test.jpg')
# # print(pic1.shape)
# transf = transforms.ToTensor()
# pic1_transf = transf(pic1)
# # print(pic1_transf.shape)
# det_out, da_seg_out,ll_seg_out = yolo(pic1_transf)
# 递归查找文件夹中的文件
import shutil
import os
testpath = "F:\postgraduate\数据集 街景天气\Copy of rainy1\\rainy"
targetpath = "F:\postgraduate\数据集 街景天气\\rainy\\"
pics = []
for root, dirs, files in os.walk(testpath):
    print("root:",root)
    # print("dirs:",dirs)
    print("files:",files)
    l = len(files)
    while(l > 0):
        src = root + '\\' +files[l-1]
        print(src)
        shutil.copy2(src,targetpath)
        l = l - 1
# 将某个文件夹中 文件名中带有关键字 的文件 复制到另一个文件夹
import os

src_dir_path = 'F:\\postgraduate\\数据集 街景天气\\sunnytest'  # 源文件夹

to_dir_path = 'F:\\postgraduate\\数据集 街景天气\\'  # 存放复制文件的文件夹

key = '3_00013'  # 源文件夹中的文件包含字符key则复制到to_dir_path文件夹中

if not os.path.exists(to_dir_path):
    print("to_dir_path not exist,so create the dir")
    os.mkdir(to_dir_path, 1)
if os.path.exists(src_dir_path):
    print("src_dir_path exist")
    for file in os.listdir(src_dir_path):
        # is file
        if os.path.isfile(src_dir_path + '/' + file):
            if key in file:
                print('找到包含"' + key + '"字符的文件,绝对路径为----->' + src_dir_path + '/' + file)
                print('复制到----->' + to_dir_path + file)
                shutil.copy(src_dir_path + '/' + file, to_dir_path + '/' + file)  # 移动用move函数
