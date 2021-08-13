#Import Required Libraries
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import UtilFuncs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import datetime
from torch.utils.tensorboard import SummaryWriter

# ###########resnet50
# #Creating Arguments for CLI
# parser = argparse.ArgumentParser(description='Training File')
# # parser.add_argument('data_dir', nargs=1, action="store", default=["./flowers"])
# parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
# # parser.add_argument('--save_dir', dest="save_dir", action="store", default="./train_model_save/checkpoint.pth")
# parser.add_argument('--learning_rate', dest="learning_rate", type=float, action="store", default=0.0001)
# parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.7)
# parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=150)
# parser.add_argument('--arch', dest="arch", action="store"   , default="resnet50", type=str)
# # parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
# parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
# pa = parser.parse_args()

###########resnet50
#Creating Arguments for CLI
parser = argparse.ArgumentParser(description='Training File')
# parser.add_argument('data_dir', nargs=1, action="store", default=["./flowers"])
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
# parser.add_argument('--save_dir', dest="save_dir", action="store", default="./train_model_save/checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", type=float, action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.4)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=100)
parser.add_argument('--arch', dest="arch", action="store"   , default="resnet50", type=str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=32)
# parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=128)
pa = parser.parse_args()


# path = pa.save_dir
lr = pa.learning_rate
structre = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


for i in range(5):

  train_annotation_path ="./train_txt/train"+str(i)+".txt"
  val_annotation_path  ="./train_txt/val"+str(i)+".txt"
  class_num = 2
  now = datetime.datetime.now()
  time_str = now.strftime("%Y-%m-%d_%H:%M:%S")

  dir_path = "./train_model_save/last"
  os.makedirs(dir_path,exist_ok=True)

  path = dir_path+"/"+time_str+structre+str(i)+"_.pth"
  log_dir = os.path.join('tensorboard','last', structre+"_"+str(i))

  os.makedirs(log_dir,exist_ok=True)

  writer = SummaryWriter(log_dir=log_dir)
  trainloader,validloader =  UtilFuncs.load_data(train_annotation_path,val_annotation_path,batch_size = 128*8)

  model,criterion,optimizer,scheduler = UtilFuncs.buildNN(structre,lr,dropout,hidden_layer1,power,epochs,class_num)
  UtilFuncs.trainNN(model, criterion, optimizer,trainloader, validloader, epochs, power,scheduler,writer)
  UtilFuncs.save_checkpoint(model,path,structre,hidden_layer1,dropout,lr,epochs)
  print("Model trained and saved successfully!!")
