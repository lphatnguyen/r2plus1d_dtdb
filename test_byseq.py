import torch
import cv2
import numpy as np
import torchvision
from dtdb_dataset import dtdb_dataset
from tqdm import tqdm

scaleFactor = [1,np.sqrt(2),2, 2*np.sqrt(2),4]

class test_database(object):
    def __init__(self,model,test_path,is_dog = True):
        self.model = model.to(torch.device('cuda:0'))
        self.test_path = test_path
        self.sequence_length = 25
        self.is_dog = is_dog
        self.test_set = dtdb_dataset(path = self.test_path,
                                     sequence_length = self.sequence_length,
                                     is_train = False,
                                     is_dog = self.is_dog,
                                     cuda_device = 'cuda:0')
    
    def get_accuracy(self):
        print('Start getting accuracy!')
        correct = 0
        nb_blocks = 0
        for i in range(len(self.test_set.lbs)):
            print('Getting accuracy for video:', i, '/', len(self.test_set.lbs), ': ', self.test_set.videos[i])
            prediction, nb = self.predict_vid(i)
            correct += prediction
            nb_blocks += nb
            print('correct = ', correct, ' out of ', nb_blocks)
        return correct/nb_blocks
    
    def predict_vid(self,idx):
        vid_name = self.test_set.videos[idx]
        ground_truth = self.test_set.lbs[idx]
        correct = 0
        cap = cv2.VideoCapture(vid_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hw_ratio = frame_height/frame_width
        wh_ratio = frame_width/frame_height
        if frame_width > 256:
            frame_width = 256
            frame_height = int(hw_ratio * frame_width)
        
        if frame_height > 256:
            frame_height = 256
            frame_width = int(wh_ratio * frame_height)
        
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps > 30:
            downsample_rate = int(vid_fps / 25)
        else: 
            downsample_rate = 1
        
        num_blocks = int(frame_count/self.sequence_length-1)
        num_blocks_nonempty = 0
        for i in tqdm(range(num_blocks)):
            if self.is_dog:
                frame_block = np.empty((0,frame_height,frame_width,5))
            else:
                frame_block = np.empty((0,frame_height,frame_width,3))
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(self.sequence_length*i))
            fc = 0
            fc_1 = 0
            ret = True
            while fc_1<self.sequence_length and ret:
                ret,frame = cap.read()
                if fc%downsample_rate == 0 and np.array(frame).all() != None:
                    frame = cv2.resize(frame,(frame_width,frame_height),interpolation = cv2.INTER_LINEAR)
                    if self.is_dog:
                        frame = self.calcDog(frame)
                    frame = np.expand_dims(frame,axis=0)
                    frame_block = np.concatenate((frame_block,frame),axis=0)
                    fc_1 += 1
                fc += 1

            frame_block = frame_block.astype(np.float32) / 255.0
            frame_block = torch.tensor(frame_block.transpose(3,0,1,2), device = 'cuda:0')
            frame_block = frame_block.unsqueeze(0)
            if frame_block.shape[2]>0:
                num_blocks_nonempty += 1
                output = self.predict_clip(frame_block)
                if output==ground_truth:
                    correct += 1
                # else:
                    # print('output=', output, ' gt=', ground_truth)
        cap.release()
        # print(correct, ' correct blocks out of', num_blocks_nonempty)
        #if num_blocks_nonempty==0:
        #    return 1
        #else:
        return correct, num_blocks_nonempty
        
    def predict_clip(self,sequence):
        assert len(sequence.size())==5, "Input sequence has to be a tensor of exactly 5 dimensions NxCxDxHxW"
        predict_vect = self.model(sequence)
        _,predict_class = torch.max(predict_vect.data,1)
        predict_class = predict_class.item() # data.tolist()
        return predict_class
    
    def calcDog(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dogImg = np.empty((image.shape[0],image.shape[1],0))
        dog1 = -cv2.GaussianBlur(image,(0,0),scaleFactor[0])+image
        dog1[dog1<0]=0
        dog1 = np.expand_dims(dog1,axis = 2)
        dogImg = np.concatenate((dogImg,dog1),axis = 2)
        for k in range(len(scaleFactor)-1):
            blur1 = cv2.GaussianBlur(image,(0,0),scaleFactor[k])
            blur2 = cv2.GaussianBlur(image,(0,0),scaleFactor[k+1])
            tempDoG = blur1 - blur2 
            tempDoG[tempDoG<0]=0
            tempDoG = np.expand_dims(tempDoG,axis = 2)
            dogImg = np.concatenate((dogImg,tempDoG),axis = 2)
        return dogImg

device = torch.device('cuda:0')

model = torchvision.models.video.r2plus1d_18(num_classes = 18).to(device)
is_dog = True
if is_dog:
    model.stem[0] = torch.nn.Conv3d(5, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
path = '../../datasets/DTDB_dynamic_based/'

state_dict = torch.load('../dtdb_dynamics_RGB_DoG/trained_models/dtdb_wts_R2p1D/dtdb_wts_R2p1D_DoG.pth', map_location='cuda:0')
model.load_state_dict(state_dict['bestModelWts'])
print('Model loaded')

test_data = test_database(model = model,
                          test_path = path,
                          is_dog = is_dog)
print('Test accuracy is:', test_data.get_accuracy())
