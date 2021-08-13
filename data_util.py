import cv2
from numpy.lib.function_base import piecewise
import pandas as pd
import random 
random.seed(10)
import os
from tqdm import tqdm
# # 65cff374-1e99-4e48-8ad5-32c909948705.png
# ddd = cv2.imread('D:/ali/v5/v5/images140721/65cff374-1e99-4e48-8ad5-32c909948705.png')

# # f6f0bb6e-d4be-4eb7-8492-cd6c3a750f5d.png


# print(ddd.shape)

# exec()


file_path =r"D:\ali\v5\v5\OneDrive_1_2021-7-20\catpainlabels140721.csv"
root_image_path = "D:/ali/v5/v5/images140721/"
image_root_path ="./images140721"


####v3
file_path ="D:/ali/v3/catpainlabels170621.csv"
root_image_path ="D:/ali/v3/v3/"
image_root_path ="./v3"
root_txt_path ="./v3_txt/train_txt"


####v1
file_path ="D:/ali/v1/catpainlabels260521.csv"
root_image_path ="D:/ali/v1/v1/"
image_root_path ="./v1"
root_txt_path ="./v1_txt/train_txt"



####v4
file_path ="D:/ali/v4/catpainlabels020721.csv"
root_image_path ="D:/ali/v4/v4/"
image_root_path ="./v4"
root_txt_path ="./v4_txt/train_txt"


os.makedirs(root_txt_path,exist_ok=True)



# df = pd.read_excel(file_path)
df = pd.read_csv(file_path)
labels_df = df['overall_impression']
label_list = []

# label_tag_dict = {0:['pain_moderately_present_(1)','pain_markedly_present_(2)','pain_present_but_unsure_how_to_grade_specifically'],1:['pain_absent_(0)']}
# label_tag_dict = {1:['pain_moderately_present_(1)'],0:['pain_absent_(0)'],3:['unable_to_assess_for_other_reason'],2:['pain_markedly_present_(2)']}#####4
label_tag_dict = {0:['pain_absent_(0)'],1:['pain_moderately_present_(1)'],2:['pain_markedly_present_(2)'],3:['unable_to_assess_for_other_reason']}####4
label_tag_dict = {0:['pain_absent_(0)'],1:['pain_moderately_present_(1)'],2:['pain_markedly_present_(2)']}####3
# label_tag_dict = {0:['pain_moderately_present_(1)'],1:['pain_absent_(0)']}####2
label_dict = {}


for index, row in df.iterrows():
  # print(row['image_id'])
  # print(row['overall_impression'])
  label = row['overall_impression']
  
  image_path =root_image_path+row['image_id']
  ddd = cv2.imread(image_path)
  if ddd is None:
    continue

  if not label in label_dict.keys():
    label_dict[label] =[]
  label_dict[label].append(row['image_id'])

max_value = 0
for key in label_dict.keys():

  tmp_value = len(label_dict[key])

  if tmp_value > max_value:
    max_value = tmp_value


for key in label_tag_dict.keys():
  
  # print(key)
  len_ = len(label_dict[label_tag_dict[key][0]])
  # print(len_)
  print(key,max_value/len_)
print(10*'=')


train_list = []
val_list = []
split_rate =0.2



five_data = []
for label_key in label_dict.keys():
  tag = None
  for label_tag in label_tag_dict.keys():
    if label_key in label_tag_dict[label_tag]:
      tag = label_tag
      break

  image_list = label_dict[label_key]

  data={'data':None,'val_splice':[]}
  if not tag is None:
    image_list = [ image_root_path +'/'+x+"*"+str(tag)+"\n" for x in image_list]
    random.shuffle(image_list)
    image_list_len = len(image_list)

    diff_num = int(image_list_len/5)
    data['data'] = image_list
    start=0
    for i in range(5):
      start_num =start*diff_num
      end_num = (start+1)*diff_num
      start +=1
      data['val_splice'].append([start_num,end_num])
      # print(start_num,end_num)
    # print(image_list_len)
    ####################
    # 划分五折
    # train_len = int(image_list_len*split_rate)
    # train_list.extend(image_list[:train_len])

    # val_list.extend(image_list[train_len:])
    five_data.append(data)

for i in range(5):
  train_list=[]
  val_list = []
  for tmp_data in five_data:
    image_list = tmp_data['data']
    split_num = tmp_data['val_splice'][i]
    start_num,end_num = split_num

    val_list.extend(image_list[start_num:end_num])
    if start_num==0:
      train_list.extend(image_list[end_num:-1])
    else:
      train_list.extend(image_list[0:start_num])
      train_list.extend(image_list[end_num:-1])

    # print(start_num,end_num)


  with open(root_txt_path+'/train'+str(i)+'.txt','w') as f:
    f.writelines(train_list)
  with open(root_txt_path+'/val'+str(i)+'.txt','w') as f:
    f.writelines(val_list)








# print(label_dict.keys())
# '''
# {
#   'pain_moderately_present_(1)': 532,
#   'pain_absent_(0)': 1656,
#   'unable_to_assess_for_other_reason': 26,
#   'pain_markedly_present_(2)': 55,
#   'pain_present_but_unsure_how_to_grade_specifically': 5  ### pain  present  but unsure  how to  grade specifically
# }

# '''

