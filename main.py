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

import numpy as np

from tqdm import tqdm

from PIL import ImageEnhance
from sklearn.metrics import confusion_matrix


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def get_random_data(raw_image,input_shape=(224, 224)):
    h, w = input_shape
    image = raw_image

    nh, nw,_  = image.shape
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


    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image



    # # resize image
    image = np.asarray(image).astype("float32") / 255
    # image = cv2.resize(image, input_shape).astype('float32') / 255
    return image


#Return Model,Criterion,Optimizer
def buildNN(arch,dropout=0.5,hidden_units=128,class_num =4):
    
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
                                    ('fc1',nn.Linear(2048,hidden_units,bias=True)),
                                    ('relu1',nn.ReLU()),
                                    ('drop1',nn.Dropout(p=dropout)),
                                    ('fc2',nn.Linear(hidden_units,hidden_units,bias=True)),
                                    ('relu2',nn.ReLU()),
                                    ('drop2',nn.Dropout(p=dropout)),####0.2
                                    ('fc3',nn.Linear(hidden_units,class_num,bias=True)),
                                    ]))
    elif(arch=="densenet121"):
        model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1',nn.Linear(1024,hidden_units,bias=True)),
                                    ('relu1',nn.ReLU()),
                                    ('drop1',nn.Dropout(p=dropout)),####0.2
                                    ('fc2',nn.Linear(hidden_units,hidden_units,bias=True)),
                                    ('relu2',nn.ReLU()),
                                    ('drop2',nn.Dropout(p=dropout)),####0.2
                                    ('fc3',nn.Linear(hidden_units,class_num,bias=True))
                                    ]))
    else:
        print("Pls try to use resnet50 or densenet121")


    return model

if __name__=='__main__':


    model_type = 'resnet50'
    # model_path=r"C:\Users\972634107\Desktop\log\resnet50\4_new_data_new_model\2021-07-27_13_39_01resnet50.pth"
    # model = buildNN(model_type,dropout=0.7,hidden_units=32,class_num =2)
    model_type = 'densenet121'
    model_path=r"C:\Users\972634107\Desktop\log\densenet1212021-07-28_17_34_37\2021-07-28_17_34_37densenet121.pth"
    model = buildNN(model_type,dropout=0.5,hidden_units=128,class_num =2)

    ############model_best
    # model_path = './train_model_save/2021-07-06 17_28_51densenet121.pth'
    # # model_path = './train_model_save/2021-06-24 21_13_17resnet50.pth'

    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    # model = buildNN(model_type,dropout=0.5,hidden_units=128,class_num =3)

    # model_path = './train_model_save/2021-07-07 18_39_03resnet50.pth'

    # model_path =r'C:\Users\972634107\Desktop\log\densenet1212021-07-15_14_42_50\2021-07-15 14_42_50densenet121.pth'


    # ######model_best


    transforms_train=transforms.Compose([
        transforms.ToTensor()
    ])  

    y_true = []
    y_pred = []
    from tqdm import tqdm
    # with torch.no_grad():

    #     with open('./val.txt','r') as f:
    #         val_data = f.readlines()

    #     val_data = [x.strip() for x in val_data]

    #     for tmp_lines in tqdm(val_data):
                
    #         tmp_list = tmp_lines.split("*")

    #         # ./catimages//62441f1a-7304-4061-a89d-3c071194361b.png*0
    #         image_path = tmp_list[0]

    #         image_path = image_path.replace("./images140721",'D:/ali/v5/v5/images140721')
    #         image_label = int(tmp_list[1])

    #         y_true.append(image_label)
    #         raw_image = cv2.imread(image_path)
    #         inputs = get_random_data(raw_image)

    #         #   inputs = np.expand_dims(inputs,axis=0)
    #         # print(inputs.shape)
    #         inputs = transforms_train(inputs)
            
    #         inputs = inputs.unsqueeze(0)
    #         output = model.forward(inputs)


    #         #   print()
    #         output = softmax(output.cpu().numpy()[0])

    #         # print(output)
    #         output = output.tolist()
            

    #         y_pred.append(output.index(max(output)))
    # confusion =confusion_matrix(y_true, y_pred)
    # print(confusion)

    # exec()
    #coding=utf-8
    import matplotlib.pyplot as plt
    import numpy as np
    


    # confusion = np.array(([0,119],[0,332]))



    # plt.imshow(confusion, cmap=plt.cm.Blues)
    # # ticks
    # # label
    # indices = range(len(confusion))
    # #
    # #plt.xticks(indices, [0, 1, 2])
    # #plt.yticks(indices, [0, 1, 2])
    # plt.xticks(indices, ['pain_moderately_present', 'pain_absent','unable_to_assess_for_other_reason','pain_markedly_present'])
    # plt.yticks(indices, ['pain_moderately_present', 'pain_absent','unable_to_assess_for_other_reason','pain_markedly_present'])

    # plt.colorbar()

    # plt.xlabel('pred')
    # plt.ylabel('true')
    # plt.title(model_type +' confusion matrix')

    # # plt.rcParams
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False


    # for first_index in range(len(confusion)):
    #     for second_index in range(len(confusion[first_index])):
    #         plt.text(first_index, second_index, confusion[first_index][second_index])
    # # 在matlab里面可以对矩阵直接imagesc(confusion)
    # # 显示
    # plt.show()
    import seaborn as sn
    import pandas as pd

    # # print(confusion)
    # confusion = np.array([[  7, 157 , 48 ,  5],[ 18 ,600 ,106,  19],[  0 ,  4 ,  8,   0],[  2,  12  , 5,   1]]) 

    # # df_cm = pd.DataFrame(confusion,, ['pain_moderately_present', 'pain_absent','unable_to_assess_for_other_reason','pain_markedly_present'])
    # # # plt.figure(figsize=(10,7))
    # # sn.set(font_scale=1.4) # for label size
    # # sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size


    # df_cm = pd.DataFrame(confusion, index = ['pain_moderately_present', 'pain_absent','unable_to_assess_for_other_reason','pain_markedly_present'],
    #               columns = ['pain_moderately_present', 'pain_absent','unable_to_assess_for_other_reason','pain_markedly_present'])
    # plt.figure(figsize = (10,7))
    # sn.heatmap(df_cm, annot=True)

    # plt.show()


    import seaborn
    import matplotlib.pyplot as plt
    
    
    def plot_confusion_matrix(data, labels, output_filename,model_type):
        """Plot confusion matrix using heatmap.
    
        Args:
            data (list of list): List of lists with confusion matrix data.
            labels (list): Labels which will be plotted across x and y axis.
            output_filename (str): Path to output file.
    
        """
        seaborn.set(color_codes=True)
        plt.figure(1, figsize=(18, 12))
    
        plt.title(model_type +' confusion matrix')
    
        seaborn.set(font_scale=1.4)
        ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='d')
    
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
        # ax.set(ylabel="", xlabel="")

        
        plt.xlabel("Predicted Label",fontsize=20,fontweight='bold')
        plt.ylabel("True Label",fontsize=20,fontweight='bold')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    # define data


    # best
    # resnet50
    data =[[863,115],[209,111]]####0.7503852080123267
    ###densenet121
    data = [[ 941,37 ],[  273,47] ]####0.7611710323574731




    # data =[[110, 182 ,  3],[134, 821 ,  0],[ 14 ,  9  , 2]]###0.731764705882353
    # data =[[121, 170 ,  4],[130, 823  , 2],[ 13 , 10  , 2]]  ###0.7419607843137255
    acc = data[0][0]+data[1][1]
    total = 0
    for i in range(2):
        for j in range(2):
            total += data[i][j]
    print(acc/total)
    # labels =  ['pain_moderately_present_(1)', 'pain_absent_(0)','pain_markedly_present_(2)']


    # data =[[821,134,    0],[182 ,110,  3],[ 9 ,  14  , 2]]###0.731764705882353
    # data =[[ 823  , 130,2],[ 170 ,121,  4],[ 10  , 13 , 2]]  ###0.7419607843137255  resnet50
    # labels =  ['pain_absent_(0)','pain_moderately_present_(1)','pain_markedly_present_(2)']
    labels =  ['pain_absent_(0)','pain_moderately_present_(1)']

    
# label_tag_dict = {0:['pain_moderately_present_(1)'],1:['pain_absent_(0)'],2:['pain_markedly_present_(2)']}####3个类别

    # pain_absent_(0)  pain_moderately_present_(1)  pain_markedly_present_(2) 
    
    # create confusion matrix
    plot_confusion_matrix(data, labels, model_type+'_'+"confusion_matrix.png",model_type)