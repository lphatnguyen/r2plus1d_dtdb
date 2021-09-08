import torch
import cv2
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import os

scaleFactor = [1,np.sqrt(2),2, 2*np.sqrt(2),4]

class dtdb_dataset(Dataset):
    def __init__(self, path, sequence_length = 25, is_train = True, is_dog = False, cuda_device = 'cuda:0'):
        super(dtdb_dataset,self).__init__()
        self.path = path
        self.is_dog = is_dog
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.cuda_device = cuda_device
        if self.is_train:
            self.path = self.path + 'TRAIN/'
        else:
            self.path = self.path + 'TEST/'
        
        self.classes = sorted(os.listdir(self.path))
        print(len(self.classes),' classes')

        self.videos = [] # sorted(glob(self.path + '*/*'))
        self.lbs = []
        i = 0
        for classe in self.classes:
            class_path = self.path + classe + '/'
            num_vids = len(os.listdir(class_path))
            self.videos = self.videos + glob(class_path+'*')
            self.lbs = self.lbs + [i]*num_vids
            i += 1
        print(len(self.videos), ' videos')
            
        self.clip_fns = []
        self.idx_frame = []
        self.clip_length = []
        self.clip_lbs = []
        self.clip_downsample_rate = []
        for i in range(len(self.videos)):
            fn = self.videos[i]
            lb = self.lbs[i]
            cap = cv2.VideoCapture(fn)
            fn_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fn_fps > 30:
                downsample_rate = int(fn_fps / 25)
            else: 
                downsample_rate = 1
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_clip = int(frame_count//(self.sequence_length*downsample_rate))
            for j in range(num_clip):
                (self.clip_downsample_rate).append(downsample_rate)
                (self.clip_fns).append(fn)
                (self.clip_lbs).append(lb)
                (self.idx_frame).append(j*self.sequence_length*downsample_rate)
                (self.clip_length).append(self.sequence_length)
            cap.release()
#            if (frame_count-self.sequence_length*(j+1))>=self.sequence_length:
#                (self.clip_fns).append(fn)
#                (self.clip_lbs).append(lb)
#                (self.idx_frame).append((j+1)*self.sequence_length)
#                (self.clip_length).append(frame_count-self.sequence_length*(j+1))
            
        
    def __len__(self):
        return len(self.clip_fns)
    
    def __getitem__(self,idx):
        fn = self.clip_fns[idx]
        lb = self.clip_lbs[idx]
        cap = cv2.VideoCapture(fn)
        cap.set(cv2.CAP_PROP_POS_FRAMES,self.idx_frame[idx])
#        fn_fps = cap.get(cv2.CAP_PROP_FPS)
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
        
        
        if self.is_dog:
            frame_block = np.empty((0,frame_height,frame_width,5),dtype=np.float32)
        else:
            frame_block = np.empty((0,frame_height,frame_width,3),dtype=np.float32)
#        if fn_fps > 30:
#            downsample_rate = int(fn_fps / 25)
#        else: 
#            downsample_rate = 1
        fc = 0
        downsample_rate = self.clip_downsample_rate[idx]
        fc_1 = 0
        while fc < frame_count and fc_1 < self.sequence_length:
            _,frame = cap.read()
            if fc%downsample_rate == 0 and np.array(frame).all() != None:
                frame = cv2.resize(frame,(frame_width,frame_height),interpolation = cv2.INTER_LINEAR)
                frame = frame.astype(np.float32)/255.0
#                print(frame.dtype)
                if self.is_dog:
                    frame = self.calcDog(frame)
                frame = np.expand_dims(frame,axis=0)
                frame_block = np.concatenate((frame_block,frame),axis=0)
                fc_1 += 1
            fc += 1
        
        cap.release()
#        print(frame_block.dtype)
        # frame_block = frame_block/255.0 # .astype(np.float32) / 255.0
        frame_block = torch.tensor(frame_block.transpose(3,0,1,2), device = self.cuda_device)
        lb = torch.tensor(lb,device = self.cuda_device)
        # print('Input size is: ', frame_block.size(),'in file:',fn)
        return frame_block,lb        
        """
        fn = self.videos[idx]
        lb = self.lbs[idx]
        """
    
    def calcDog(self,image):
        image_rgb = image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dogImg = np.empty((image.shape[0],image.shape[1],0),dtype = np.float32)
        dog1 = -cv2.GaussianBlur(image,(0,0),scaleFactor[0])+image
        dog1[dog1<0]=0
        dog1 = np.expand_dims(dog1,axis = 2)
        # dogImg = np.concatenate((dogImg,image_rgb),axis = 2)
        dogImg = np.concatenate((dogImg,dog1),axis = 2)
        for k in range(len(scaleFactor)-1):
            blur1 = cv2.GaussianBlur(image,(0,0),scaleFactor[k])
            blur2 = cv2.GaussianBlur(image,(0,0),scaleFactor[k+1])
            tempDoG = blur1 - blur2 
            tempDoG[tempDoG<0]=0
            tempDoG = np.expand_dims(tempDoG,axis = 2)
            dogImg = np.concatenate((dogImg,tempDoG),axis = 2)
        return dogImg
    
def watching_vid(np_block):
    for i in range(np_block.shape[0]):
        cv2.imshow('Video',np_block[i,:,:,:])
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    path = '../../flowDeep/datasets/DTDB/'
    import time
    dataset = dtdb_dataset(path,sequence_length=25,cuda_device='cuda')
#    a,b = dataset.__getitem__(394)
    dataloader = torch.utils.data.DataLoader(dataset)
    time0 = time.time()
    for i,(a,b) in enumerate(dataloader):
        x = 0
    print('Elapsed time : %.3f seconds' %(time.time()-time0))
#    shortest_length = dataset.shortest_length
#    longest_length = dataset.longest_length
#    highest_height = dataset.highest_frame
#    shortest_height = dataset.shortest_frame
#    largest_width = dataset.largest_frame
#    smallest_width = dataset.smallest_frame
