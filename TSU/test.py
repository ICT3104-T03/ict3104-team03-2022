from __future__ import division
import time
import os
import argparse
import sys
import cv2
import csv
import math

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)', default='rgb')  # set default to rgb
parser.add_argument('-train', type=str2bool, default='False', help='train or eval')  # Train true to false
parser.add_argument('-comp_info', type=str, default='TSU_CS_RGB_PDAN') # Default set to TSU_CS_RGB_PDAN
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='3')
parser.add_argument('-dataset', type=str, default='TSU')  # Changed charades to TSU
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.0002')  # Default changed 0.1 to 0.0002
parser.add_argument('-epoch', type=str, default='50')  # Default changed 50 to 140
parser.add_argument('-model', type=str, default='PDAN')  # Default set to PDAN
parser.add_argument('-APtype', type=str, default='map')  # Change wap to map
parser.add_argument('-randomseed', type=str, default='False')
# parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-load_model', type=str, default='./TSU/models/PDAN_TSU_RGB')
parser.add_argument('-num_channel', type=str, default='512')  # Change false to 512
parser.add_argument('-batch_size', type=str, default='1')  # Change false to 2/1
parser.add_argument('-kernelsize', type=str, default='2')  # Change false to 3/2
parser.add_argument('-feat', type=str, default='False')
parser.add_argument('-split_setting', type=str, default='CS')  # Default set to CS
### for inferencing, only load 1 video
parser.add_argument('-video_name', type=str)
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# set random seed
if args.randomseed=="False":
    SEED = 0
elif args.randomseed=="True":
    SEED = random.randint(1, 100000)
else:
    SEED = int(args.randomseed)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED!!!:', SEED)

from torch.optim import lr_scheduler
from torch.autograd import Variable

import json

import pickle
import math



if str(args.APtype) == 'map':
    from apmeter import APMeter


batch_size = int(args.batch_size)

if args.dataset == 'TSU':
    split_setting=str(args.split_setting)
    
    from smarthome_i3d_per_video import TSU as Dataset
    from smarthome_i3d_per_video import TSU_collate_fn as collate_fn
    classes=51

    # Uncomment this if you're using the IDE version
    ### Removed the path for train_split as we are not taking any data for training, just need to test
    if split_setting =='CS':
        train_split = ''
        test_split = './TSU/tsu_data/smarthome_CS_51.json'
        
    elif split_setting =='CV':
        train_split = ''
        test_split = './TSU/tsu_data/smarthome_CV_51.json'
    
    rgb_root = './TSU/TSU_RGB_i3d_feat/RGB_i3d_16frames_64000_SSD'
    skeleton_root='/skeleton/feat/Path/' # 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_data_rgb_skeleton(train_split, val_split, root_skeleton, root_rgb):
    # Load Data
   
    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root_skeleton, root_rgb, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn) # 8
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root_skeleton, root_rgb, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn) #2

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


def load_data(train_split, val_split, root):
    # Load Data
    print("Length of train_split", len(train_split))
    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    since = time.time()

    best_map = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map, indexArray, classAccuracy = val_step(model, gpu, dataloader['val'], epoch)
            probs.append(prob_val)
            sched.step(val_loss)

            if best_map < val_map:
                best_map = val_map
                torch.save(model.state_dict(),'./TSU/'+str(args.model)+'/weight_epoch_'+str(args.lr)+'_'+str(epoch))
                torch.save(model,'./TSU/'+str(args.model)+'/model_epoch_'+str(args.lr)+'_'+str(epoch))
                print('save here:','./TSU/'+str(args.model)+'/weight_epoch_'+str(args.lr)+'_'+str(epoch))

def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, :int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)
    
    outputs_final = activation

    if args.model=="PDAN":
        outputs_final = outputs_final[:,0,:,:]

    outputs_final = outputs_final.permute(0, 2, 1)  
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)  

    loss = loss_f 

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data
        loss.backward()
        optimizer.step()
    if args.APtype == 'wap':
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    print('train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        # print("data", data[0])
        num_iter += 1
        other = data[3]       

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probsTest = probs.data.cpu().numpy()
        indexArray = []
        for i in range(1, len(probsTest[0])):
            # print(np.argmax(probsTest[0][i]))
            indexArray.append(np.argmax(probsTest[0][i]))

        # print(indexArray)
        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print("apm value: ",100 * apm.value())
    classAccuracy = 100 * apm.value()
    apm.reset()
    return full_probs, epoch_loss, val_map, indexArray, classAccuracy

"""
description: calculate total number of frames for a single mp4 video
:param videoName    - specify name of the video 
:return length      - total number of frames for that video
"""
def calculate_frames(videoName):
    cap = cv2.VideoCapture("./data/rgbVideos/{fn}".format(fn=videoName))
    if not cap.isOpened(): 
        print("could not open :{fn}".format(fn=videoName))
        return
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


"""
description: remove unnecessary decimal points for a long decimal input
:param n        - number to be truncated
:param decimals - decimal point to be truncated to
:return number  - truncated number 
"""
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

"""
description: retrieve the output's action index array, map it with the action list, 
             generate time frame for each action and store it into a csv file.
:param indexArray   - array of action indexes retrieved from val_step function
:param frames       - total number of frames for the video 
:param video_name    - to store the csv file with the video name
:param model_name    - to store the csv file with the model name
:param val_map      - accuracy of the validation mapping
:param val_loss     - accuracy for validation loss
:param class_accuracy- accuracy per actions in a tensor array
:return -
"""
def output_csv(indexArray, frames, video_name, model_name, val_map, val_loss, class_accuracy):
    model = model_name.replace("./TSU/models/", "")
    video = video_name.replace(".mp4", "")

    header = ['action', 'start','end']
    file = open('./data/generatedAnnotations/{model_name}_{video_name}.csv'.format(model_name=model, video_name=video), 'w', encoding='UTF8', newline='')
    actionList = ["Enter", "Walk","Make_coffee", "Get_water", "Make_coffee", "Use_Drawer", "Make_coffee.Pour_grains", 
                    "Use_telephone", "Leave", "Put_something_on_table", "Take_something_off_table" , "Pour.From_kettle", 
                    "Stir_coffee/tea", "Drink.From_cup", "Dump_in_trash", "Make_tea", "Make_tea.Boil_water", "Use_cupboard",
                    "Make_tea.Insert_tea_bag" , "Read", "Take_pills", "Use_fridge", "Clean_dishes", "Clean_dishes.Put_something_in_sink",
                    "Eat_snack", "Sit_down", "Watch_TV", "Use_laptop", "Get_up", "Drink.From_bottle", "Pour.From_bottle",
                    "Drink.From_glass", "Lay_down", "Drink.From_can", "Write", "Breakfast", "Breakfast.Spread_jam_or_butter",
                    "Breakfast.Cut_bread", "Breakfast.Eat_at_table", "Breakfast.Take_ham", "Clean_dishes.Dry_up", "Wipe_table",
                    "Cook", "Cook.Cut", "Cook.Use_stove", "Cook.Stir", "Cook.Use_oven", "Clean_dishes.Clean_with_water",
                    "Use_tablet", "Use_glasses", "Pour.From_can"]
    writer = csv.writer(file)
    writer.writerow(["Validation Mapping Accuracy: ", val_map.item()])
    writer.writerow([])
    writer.writerow(["Validation Loss: ", val_loss.item()])
    writer.writerow([])
    writer.writerow(["Overall Accuracy per action"])
    for i in range(0, len(class_accuracy.numpy())):
        if (class_accuracy.numpy()[i] > 0):
            writer.writerow([actionList[i], class_accuracy.numpy()[i]])
    
    writer.writerow([])

    writer.writerow(header)
    currentFrames = 0
    endFrames = 0
    print(frames/len(indexArray))
    framesPerIndex = frames/len(indexArray)
    
    for i in range(0, len(indexArray)):
        if ((i < len(indexArray)-1) and (indexArray[i] == indexArray[i+1]) ):
            endFrames += framesPerIndex
        else:
            endFrames += framesPerIndex
            writer.writerow([actionList[indexArray[i]], truncate(currentFrames), truncate(endFrames)])
            currentFrames = endFrames

    file.close()
    print("successfully generated annotations {model_name}_{video_name}.csv and saved to ./data/generatedAnnotations/".format(model_name=model, video_name=video))
    return

if __name__ == '__main__':
    __spec__ = None
    print(str(args.model))
    print('batch_size:', batch_size)
    print('cuda_avail', torch.cuda.is_available())

    if args.mode == 'flow': # We did not touch flow
        print('flow mode')
        # print('flow mode', flow_root)
        # dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'skeleton':
        print('Pose mode', skeleton_root)
        dataloaders, datasets = load_data(train_split, test_split, skeleton_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)

    if args.train:
        num_channel = args.num_channel
        if args.mode == 'skeleton':
            input_channnel = 256
        else:
            input_channnel = 1024

        num_classes = classes
        mid_channel=int(args.num_channel)


        if args.model=="PDAN":
            print("you are processing PDAN")
            from models import PDAN as Net
            model = Net(num_stages=1, num_layers=5, num_f_maps=mid_channel, dim=input_channnel, num_classes=classes)


        model=torch.nn.DataParallel(model)

        if args.load_model!= "False":
            # entire model
            model = torch.load(args.load_model)
            # weight
            # model.load_state_dict(torch.load(str(args.load_model)))
            print("loaded",args.load_model)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('num_channel:', num_channel, 'input_channnel:', input_channnel,'num_classes:', num_classes)
        model.cuda()

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        print(lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        run([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))
    else:
        model = torch.load(args.load_model)
        print("loaded",args.load_model)
        prob_val, val_loss, val_map, indexArray, classAccuracy = val_step(model, 0, dataloaders['val'], args.epoch)
        frames = calculate_frames(args.video_name)
        output_csv(indexArray, frames, args.video_name, args.load_model, val_map, val_loss, classAccuracy)