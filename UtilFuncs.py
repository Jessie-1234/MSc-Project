#Import Required Libraries
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import torch.optim as optim
from torch.utils import data
import math
import random
import cv2

from tqdm import tqdm
from focal_loss import focal_loss
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from PIL import ImageEnhance

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def check_val(data, max_val):
    if data < 0: data = 0
    if data > max_val: data = max_val
    return data

def get_scale(jitter, val):
    tmp = int(rand(0, jitter) * val)
    return tmp

def get_lines_separate(train_annotation_path, val_annotation_path, prev_str):
    with open(train_annotation_path) as f1:
        train_line = f1.readlines()
        train_line = [c.strip() for c in train_line]
    train_line = [prev_str + l for l in train_line]
    with open(val_annotation_path) as f2:
        val_line = f2.readlines()
        val_line = [c.strip() for c in val_line]
    val_line = [prev_str + a for a in val_line]

    np.random.seed(10101)
    np.random.shuffle(train_line)
    np.random.shuffle(val_line)
    np.random.seed(None)
    return train_line, val_line

def get_random_data(raw_image,input_shape=(224, 224), jitter=0.05,enhance_bool =True):
    ih, iw,_ = raw_image.shape
    h, w = input_shape
    if enhance_bool:
        ##################随机抖动
        aw, ah = iw, ih
        x1 = get_scale(jitter, aw)
        x2 = iw - get_scale(jitter, aw)
        y1 = get_scale(jitter, ah)
        y2 = ih - get_scale(jitter, ah)


    # x1, y1, x2, y2 = check_val(rx1 + get_scale(jitter, aw), iw), check_val(ry1 + get_scale(jitter, ah), ih), \
    #                  check_val(rx2 + get_scale(jitter, aw), iw), check_val(ry2 + get_scale(jitter, ah), ih)


        image = raw_image[y1:y2,x1:x2]
    else:
        image = raw_image
        
    nh, nw,_  = image.shape
    # nw, nh = image.size
    if nh ==0:
        print(10*'err')
        return None,False

    new_ar = nw/nh 
    if new_ar > 1:
        rw = int(w)
        rh = int(nh * h / nw)
    else:
        rh = int(h)
        rw = int(nw * w / nh)

    dx = int((w - rw) / 2)
    dy = int((h - rh) / 2)

    cv_resize_img = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(cv2.cvtColor(cv_resize_img, cv2.COLOR_BGR2RGB))
    # image = image.resize((rw, rh), Image.BICUBIC)
    # imgs = np.asarray(image)
    # cv2.imwrite("paste_image.jpg", imgs)

    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image
    
    # ###imgs = np.asarray(image)
    # ##cv2.imwrite("paste_image1.jpg", imgs)

    if enhance_bool:
        ##flip image
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # brightness
        enh_bri = ImageEnhance.Brightness(image)
        image = enh_bri.enhance(rand(0.5, 1.5))

        # color
        enh_col = ImageEnhance.Color(image)
        image = enh_col.enhance(rand(0.5, 1.5))

        enh_con = ImageEnhance.Contrast(image)
        image = enh_con.enhance(rand(0.5, 1.5))

        enh_sha = ImageEnhance.Sharpness(image)
        image = enh_sha.enhance(rand(0.5, 1.5))

    # # resize image
    # image.show()
    image = np.asarray(image).astype("float32") / 255
    # image = cv2.resize(image, input_shape).astype('float32') / 255
    return image



class Image_Dataset(data.Dataset):
  def __init__(self,annotation_lines,transform=None,is_test=False,is_cache=False,is_cache_num=2000,enhance_bool=True):
      super().__init__()
      self.annotation_lines=annotation_lines
      self.transform=transform

      self.is_cache=is_cache
      self.is_cache_num=is_cache_num

      self.image_cache = {}
      self.total_cache_num = 0

      self.enhance_bool = enhance_bool

  def __len__(self):
      return len(self.annotation_lines)

  def __getitem__(self, idex):
      image_path,label = self.annotation_lines[idex].split("*")
    #   print(image_path)
      if image_path in self.image_cache.keys():
        raw_image = self.image_cache[image_path]
      else:
        raw_image = cv2.imread(image_path)
        self.image_cache[image_path] = raw_image

      resize_data = get_random_data(raw_image,enhance_bool=self.enhance_bool)

      if self.transform is not None:
          resize_data=self.transform(resize_data)

      return resize_data, torch.tensor(int(label))




def load_data(train_annotation_path,val_annotation_path,batch_size=1):
    transforms_train=transforms.Compose([
        transforms.ToTensor()
    ])  
    prev_str = ""
    train_line, val_line = get_lines_separate(train_annotation_path, val_annotation_path, prev_str)

    train_data=Image_Dataset(train_line,transform=transforms_train,enhance_bool=True) 
    val_data=Image_Dataset(val_line,transform=transforms_train,enhance_bool=False)
    trainloader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data,batch_size = batch_size)

    return trainloader,validloader


#Return Model,Criterion,Optimizer
def buildNN(arch,lr=0.001,dropout=0.5,hidden_units=4096,mode='gpu',num_epochs=120,class_num =16):
    
    if(arch=='resnet50'):
        model = models.resnet50(pretrained=True)
    elif(arch=='densenet121'):
        model = models.densenet121(pretrained=True)
    
    else:
        print("Pls choose vgg16 or densenet121, Other networks are not available!")

    for param in model.parameters():
      param.requires_grad = False



    # for param in model.parameters():
    # param.requires_grad = True
    if(arch=="resnet50"):
        model.fc = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(2048, hidden_units,bias=True)),
                                    ('relu1', nn.ReLU()),
                                    ('drop1', nn.Dropout(p=dropout)),
                                    ('fc2', nn.Linear(hidden_units, hidden_units,bias=True)),
                                    ('relu2', nn.ReLU()),
                                    ('drop2', nn.Dropout(p=dropout)),
                                    ('fc3', nn.Linear(hidden_units, class_num, bias=True)),
                                    ]))
    elif(arch=="densenet121"):
        model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(1024, hidden_units, bias=True)),
                                    ('relu1', nn.ReLU()),
                                    ('drop1', nn.Dropout(p=dropout)),
                                    ('fc2', nn.Linear(hidden_units, hidden_units, bias=True)),
                                    ('relu2', nn.ReLU()),
                                    ('drop2', nn.Dropout(p=dropout)),
                                    ('fc3', nn.Linear(hidden_units, class_num, bias=True))
                                    ]))
    else:
        print("Pls try to use resnet50 or densenet121")


    if arch=="densenet121":

        freeze_layer_bool = False
        for name, value in model.named_parameters():
            if 'denseblock4' in name:
                freeze_layer_bool = True
            if freeze_layer_bool:
                value.requires_grad = True
            else:
                value.requires_grad = False

    else:
        freeze_layer_bool = False
        for name, value in model.named_parameters():
            if 'layer4' in name:
                freeze_layer_bool = True
            if freeze_layer_bool:
                value.requires_grad = True
            else:
                value.requires_grad = False

    
    # criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = focal_loss(alpha=[3.428439519852262,1,66.30357142857143,38.27835051546392], gamma=2, num_classes=class_num)
    # criterion = focal_loss(alpha=[3.238968092328581,1.0,70.16176470588235,38.78861788617886], gamma=2, num_classes=class_num)
    # criterion = focal_loss(alpha=[3.238968092328581,1.0,38.78861788617886], gamma=2, num_classes=class_num)

    criterion = focal_loss(alpha=[1.0,3.11,30.109,63.692], gamma=2, num_classes=class_num) ###v1

    # criterion = focal_loss(alpha=[1.0,3.428,38.278,66.303], gamma=2, num_classes=class_num) ###v3

    # criterion = focal_loss(alpha=[1.0,3.238,38.788], gamma=2, num_classes=class_num) ###v4

    # criterion = focal_loss(alpha=[3,1], gamma=2, num_classes=class_num) ###v5
    params = filter(lambda p: p.requires_grad, model.parameters())
    # t=int(num_epochs/5)#warmup
    # T=num_epochs
    # n_t=0.5
    optimizer = optim.Adam(params, lr=lr,weight_decay=0)

    # lambda1 = lambda epoch: (0.9*epoch / t+0.001) if epoch < t else  0.001  if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.001 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)

    # optimizer = optim.Adam(params, lr)
    
    if torch.cuda.is_available() and mode == 'gpu':
        model.cuda()
        
    return model, criterion, optimizer ,scheduler

#Validate Model
def validation(model, testloader, criterion,mode='gpu'):
    val_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        if torch.cuda.is_available() and mode=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        val_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])

        # print(equality)
        # print(equality.type(torch.FloatTensor).mean())
        accuracy += equality.type(torch.FloatTensor).mean()
    return val_loss, accuracy
    
#Train our Neural Network
def trainNN(model,criterion,optimizer,loader1, loader2, epochs=5,mode='gpu',scheduler=None,writer =None):
    steps=0
    running_loss=0
    print("----------------Training Started------------------\n")
    for e in range(epochs):
        accuracy_train = 0
        for inputs,labels in loader1 :
            steps+=1
            if torch.cuda.is_available() and mode=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            #Forwara and Backward Propogation
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()

            ps = torch.exp(logps)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy_train += equality.type(torch.FloatTensor).mean()

            running_loss += loss.item()
        scheduler.step()

        valid_loss = 0
        accuracy = 0
        model.eval()

        #VALIDATION
        with torch.no_grad():
            valid_loss, accuracy = validation(model, loader2, criterion)



        writer.add_scalar('Train/Loss', running_loss/len(loader1),e+1)
        writer.add_scalar('Train/Acc',accuracy_train/len(loader1),e+1)
        writer.add_scalar('Validation/Loss',valid_loss/len(loader2),e+1)
        writer.add_scalar('Validation/Acc',accuracy/len(loader2),e+1)

    
        print(f"Epoch {e+1}/{epochs}.. "
                f"Training loss: {running_loss/len(loader1):.3f}.. "
                f"Training accuracy: {accuracy_train/len(loader1):.3f}  "
                f"Validation loss: {valid_loss/len(loader2):.3f}.. "
                f"Validation accuracy: {accuracy/len(loader2):.3f}")
        running_loss = 0
        model.train()

    writer.close()
    print("\n---------------Training Completed---------------")

#Save Checkpoint
def save_checkpoint(model ,path='./checkpoint.pth',structure ='resnet50', hidden_layer1=4096,dropout=0.5,lr=0.001,epochs=5):
    
    model.cpu
    # model.class_to_idx = train_data.class_to_idx
    chpt = {'structure' :structure,
            'hidden_layer1':hidden_layer1,
            'dropout':dropout,
            'lr':lr,
            'no_of_epochs':epochs,
            'state_dict':model.state_dict(),
            # 'class_to_idx':model.class_to_idx
            }
    
    if structure=="resnet50":
        chpt['fc'] = model.fc
    else:
        chpt['classifier'] = model.classifier
    # if(path!="./checkpoint.pth"):
    #     path = path + "/checkpoint.pth"
    torch.save(chpt,path)
    print("Model Saved")

#Load Checkpoint
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    model,_,_,_ = buildNN(structure,lr,dropout,hidden_layer1,class_num=6)
    # model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
   
#Process PIL Image
def process_image(image_path):


    transforms_train=transforms.Compose([
        transforms.ToTensor()
    ]) 
    image_origin = cv2.imread(image_path)

    raw_image=cv2.cvtColor(image_origin,cv2.COLOR_BGR2GRAY)
    fft_fig1 = np.fft.fft2(raw_image)
    fig1_amp = np.abs(fft_fig1)
    fig1_pha = np.angle(fft_fig1)
    fig1_amp = fig1_amp.astype(raw_image.dtype)
    fig1_pha = fig1_pha.astype(raw_image.dtype)
    image_merge =cv2.merge([raw_image,fig1_amp,fig1_pha])

    resize_data = get_random_data(image_merge)

    if transforms_train is not None:
        resize_data=transforms_train(resize_data)

    return resize_data,image_merge,image_origin

#Make Prediction
def predict(image_path, model, topk=5,power='gpu'):
    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda:0')
    model.eval()
    img,_ = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img)
    probability = torch.exp(output)
    return probability
    
