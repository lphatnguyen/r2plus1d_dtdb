import torch
from dtdb_dataset import dtdb_dataset
# from C3D import C3D
import torchvision
import numpy as np
import argparse
import copy
import time
from tqdm import tqdm
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('--nbEpochs', type = int, default = 20)
parser.add_argument('--model', type = str, default = 'r2plus1d')
parser.add_argument('--is_dog', type = bool, default = True)
parser.add_argument('--sequence_length', type = int, default = 25)
parser.add_argument('--train_batch_size', type = int, default = 1)
parser.add_argument('--val_batch_size', type = int, default = 1)
parser.add_argument('--is_train', type = bool, default = True)
parser.add_argument('--cuda_device', type = str, default = 'cuda:0')
parser.add_argument('--milestones', type = int, default = [75,90], help = "Milestone for decreasing learning rates by half")
args = parser.parse_args()

print('is_dog =', args.is_dog)

def train_model(model, dataloaders, criterion, optimizer, numEpochs):
    bestModelWts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(numEpochs):
        time0 = time.time()
        print('Epoch {}/{}'.format(epoch+1, numEpochs))
        print('-' * 10)
        params = optimizer.state_dict()['param_groups'][0]
        print('Optimizer leaning rate is: ', params['lr'])
        for phase in ['train', 'val']:
            runningLoss = 0.0
            corrects = 0
            
            for i,(inputs,lbs) in tqdm(enumerate(dataloaders[phase])):
                optimizer.zero_grad()
                # print(inputs.type())
                with torch.set_grad_enabled(phase == 'train'):
                    # print('Input shape', inputs.size())
                    outputs = model(inputs)
                    loss = criterion(outputs,lbs)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    runningLoss = runningLoss + loss.item()
                
                if phase == 'val':
                    _,preds = torch.max(outputs.data,1)
                    corrects += torch.sum(preds.data == lbs.data)
                    del preds
            
            if phase == 'train':
                epochLoss = runningLoss/len(dataloaders['train'].sampler)
            else:
                epoch_acc = 1.0*corrects.tolist()/dataloaders['val'].__len__()
            
            if phase == 'val' and epoch_acc>best_acc:
                best_acc = epoch_acc
                bestModelWts = copy.deepcopy(model.state_dict())
                torch.save({'bestEpoch': epoch,
                            'bestAcc': best_acc,
                            'bestModelWts': bestModelWts},'./trained_models/dtdb_wts_R2p1D_RGB.pth')
            del outputs,loss
            
        model_wts = copy.deepcopy(model.state_dict())
        optim_wts = copy.deepcopy(optimizer.state_dict())
        torch.save({'epoch':epoch,
                    'accuracy': epoch_acc,
                    'optimWts': optim_wts,
                    'modelWts':model_wts},'./trained_models/checkpointEachIter.pth.tar')
            
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train loss and validation accuracy', epochLoss, epoch_acc))
        print('The running time for this epoch is: ', time.time()-time0)
    
    print('Best val Epe: {:4f}'.format(best_acc))
    model.load_state_dict(bestModelWts)
    return model

def test_model(model,dataloader):
    print('Begin testing process!')
    correct = 0
    accuracy = 0.0
    for i,(inputs,lbs) in enumerate(dataloader):
        pred_output = model(inputs)
        _,pred = torch.max(pred_output.data,1)
        print('i=', pred.data, ' ', lbs.data) 
        correct += (torch.sum(pred.data == lbs.data)).data
        del pred,pred_output
    accuracy = int(correct.data.cpu().numpy()) / (i+1)
    return accuracy

device = torch.device(args.cuda_device)
#if args.model == 'r2plus1d':
model = torchvision.models.video.r2plus1d_18(num_classes=18, pretrained = False)
if args.is_dog==True:
    model.stem[0] = torch.nn.Conv3d(5, 45, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
# elif args.model == 'c3d':
#    model = C3D(num_classes=45,pretrained=False, is_dog = args.is_dog)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
criterion = torch.nn.CrossEntropyLoss()

data_path = '../../datasets/DTDB/BY_DYNAMIC_FINAL/'

nbEpochs = args.nbEpochs
is_train = True # args.is_train

if is_train:
    dataset = dtdb_dataset(path = data_path,
                           sequence_length=args.sequence_length,
                           is_train = True,
                           is_dog = args.is_dog,
                           cuda_device=args.cuda_device)
    print('Dataset initialized')

    generateSplit = False
    if generateSplit:
        num_data = len(dataset)
        indices = list(range(num_data))
        idx_split = int(np.floor(0.9*num_data))
        np.random.shuffle(indices)

        num_train = indices[:idx_split]
        num_val = indices[idx_split:]

        split_train_val = {'train' : num_train, 'val' : num_val}
        torch.save(split_train_val, './split_train_val.pt')
        print('Split train/val saved')
    else:
        split_train_val = torch.load('./train_val_split.pt')
        num_train = split_train_val['num_train']
        num_val = split_train_val['num_val']
        print(len(num_train), ' training examples')
        print(len(num_val), ' validation examples')

    train_sampler = torch.utils.data.SubsetRandomSampler(num_train)
    val_sampler = torch.utils.data.SubsetRandomSampler(num_val)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size = args.train_batch_size,
                                               sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.val_batch_size,
                                             sampler=val_sampler)
    
    dataloaders = {'train': train_loader, 'val': val_loader}

    startEpoch = 0
    # if path.isfile('./trained_models/checkpointEachIter.pth.tar'):
    # 	print('Found existing model in ./trained_models/checkpointEachIter.pth.tar')
    # 	file = torch.load('./trained_models/checkpointEachIter.pth.tar', map_location='cuda:0')
    # 	startEpoch = file['epoch']
    # 	print('Model was trained until epoch ', startEpoch)
    # 	model.load_state_dict(file['modelWts'])
    # else:
    # print('Initialize a new model for training!')
    
    print('The training set contains ', train_loader.__len__(), 'instances!')
    print('The validation set contains ', val_loader.__len__(), 'instances!')
    
    model = train_model(model = model,
                        dataloaders = dataloaders,
                        criterion = criterion,
                        optimizer = optimizer,
                        numEpochs = nbEpochs-startEpoch)

# state_dict = torch.load('./dtdb_wts_R2p1D_RGB.pth',map_location='cuda:0')
# print('Model loaded')

# model.load_state_dict(state_dict['bestModelWts'])
# model.to(device)
# model = model.eval()
# test_set = dtdb_dataset(data_path,
#                         sequence_length=args.sequence_length,
#                         is_train=False,
#                         is_dog=True,
#                         cuda_device=args.cuda_device)
# print('Dataset initialized')
# test_loader = torch.utils.data.DataLoader(test_set)
# acc = test_model(model,test_loader)
# print('The final accuracy is: ', acc)
